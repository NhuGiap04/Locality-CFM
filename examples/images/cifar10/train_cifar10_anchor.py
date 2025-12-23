# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import torch
import wandb
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchvision import datasets, transforms
from tqdm import trange
from utils_cifar import ema, generate_samples, infiniteloop, ClassConditionedSampler, visualize_clusters

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_float("lambda_anchor", 1.0, help="weight for anchor regularization term")
flags.DEFINE_string("lambda_name", "1e0", help="Lambda value as string for logging")
flags.DEFINE_integer("centroid_update_freq", 1000, help="frequency of updating class centroids")
flags.DEFINE_string("anchor_loss_type", "full", help="type of anchor loss: 'full' or 'simple'")
flags.DEFINE_integer("num_subclasses", 25, help="number of subclasses for class-conditioned sampling")

# Checkpoint
flags.DEFINE_string("checkpoint_path", "", help="path to checkpoint file to resume training from")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)

flags.DEFINE_bool("visualize", True, help="visualize samples during training")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def load_state_dict_flexible(model, state_dict, is_parallel):
    """
    Load state dict handling mismatch between single GPU and DataParallel checkpoints.
    
    Parameters
    ----------
    model:
        The model to load weights into
    state_dict: dict
        The state dictionary from checkpoint
    is_parallel: bool
        Whether the current model is wrapped in DataParallel
    """
    # Check if checkpoint was saved with DataParallel (keys start with 'module.')
    checkpoint_is_parallel = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_parallel and not checkpoint_is_parallel:
        # Current model is parallel, but checkpoint is from single GPU
        # Add 'module.' prefix to all keys
        new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("Loaded single GPU checkpoint into DataParallel model")
    elif not is_parallel and checkpoint_is_parallel:
        # Current model is single GPU, but checkpoint is from DataParallel
        # Remove 'module.' prefix from all keys
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("Loaded DataParallel checkpoint into single GPU model")
    else:
        # Both match, load directly
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint directly ({'DataParallel' if is_parallel else 'single GPU'} mode)")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def compute_fid_score(model, parallel, num_gen=10000, batch_size_fid=512, integration_steps=100):
    """Compute FID score for generated samples.
    
    Parameters
    ----------
    model:
        The neural network model to generate samples from
    parallel: bool
        Whether the model is in parallel mode
    num_gen: int
        Number of samples to generate for FID computation
    batch_size_fid: int
        Batch size for FID computation
    integration_steps: int
        Number of steps for ODE integration
        
    Returns
    -------
    float
        The computed FID score
    """
    model.eval()
    
    model_ = copy.deepcopy(model)
    if parallel:
        model_ = model_.module.to(device)
    
    def gen_1_img(unused_latent):
        with torch.no_grad():
            x = torch.randn(batch_size_fid, 3, 32, 32, device=device)
            t_span = torch.linspace(0, 1, 2, device=device)
            traj = odeint(
                model_,
                x,
                t_span,
                rtol=1e-5,
                atol=1e-5,
                method="dopri5",
            )
            traj = traj[-1, :]
            img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
        return img
    
    try:
        score = fid.compute_fid(
            gen=gen_1_img,
            dataset_name="cifar10",
            batch_size=batch_size_fid,
            dataset_res=32,
            num_gen=num_gen,
            dataset_split="train",
            mode="legacy_tensorflow",
        )
    except Exception as e:
        print(f"Error computing FID: {e}")
        score = None
    
    model.train()
    return score


def compute_class_centroids(dataset, class_sampler, device='cuda'):
    """
    Compute the centroid (mean) for each class in the new class system.
    
    Uses the class_to_indices mapping from ClassConditionedSampler to compute
    centroids based on the clustered classes (after K-means).
    
    Args:
        dataset: PyTorch dataset with (image, label) pairs
        class_sampler: ClassConditionedSampler instance with class_to_indices mapping
        device: Device to store centroids
    
    Returns:
        centroids: Tensor of shape (num_classes, C, H, W) where num_classes is the
                  total number of classes after clustering
    """
    print(f"Computing class centroids for {class_sampler.num_classes} classes...")
    
    # Compute mean for each class using the sampler's class_to_indices
    centroids = []
    for class_id in sorted(class_sampler.class_to_indices.keys()):
        class_indices = class_sampler.class_to_indices[class_id]
        
        # Collect samples for this class
        class_samples = []
        for idx in class_indices[:500]:  # Limit to 500 samples per class for efficiency
            img, _ = dataset[idx.item()]
            class_samples.append(img)
        
        if len(class_samples) > 0:
            class_tensor = torch.stack(class_samples)
            centroid = class_tensor.mean(dim=0)
            centroids.append(centroid)
        else:
            # If no samples, use zeros (should not happen)
            print(f"Warning: Class {class_id} has no samples!")
            centroids.append(torch.zeros(3, 32, 32))
    
    centroids = torch.stack(centroids).to(device)
    print(f"Centroids computed with shape: {centroids.shape}")
    return centroids


def train(argv):
    # Initialize wandb
    wandb.init(
        project="conditional-flow-matching",
        name=f"{FLAGS.model}_cifar10_anchor_{FLAGS.lambda_name}",
        config={
            "model": FLAGS.model,
            "lr": FLAGS.lr,
            "total_steps": FLAGS.total_steps,
            "batch_size": FLAGS.batch_size,
            "ema_decay": FLAGS.ema_decay,
            "num_channel": FLAGS.num_channel,
            "grad_clip": FLAGS.grad_clip,
            "warmup": FLAGS.warmup,
            "lambda_anchor": FLAGS.lambda_anchor,
            "lambda_name": FLAGS.lambda_name,
            "centroid_update_freq": FLAGS.centroid_update_freq,
        },
    )
    
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    
    # Create the class-conditioned sampler first (performs clustering)
    class_sampler = ClassConditionedSampler(dataset, FLAGS.batch_size, num_classes=10, num_subclasses=FLAGS.num_subclasses)

    if FLAGS.visualize:
        print("Visualizing class clusters...")
        visualize_clusters(class_sampler, dataset)
    
    # Compute class centroids based on the new class system (after clustering)
    class_centroids = compute_class_centroids(dataset, class_sampler, device=device)
    
    dataloader_iter = iter(class_sampler)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # Load checkpoint if provided
    start_step = 0
    if FLAGS.checkpoint_path:
        print(f"Loading checkpoint from {FLAGS.checkpoint_path}")
        checkpoint = torch.load(FLAGS.checkpoint_path, map_location=device)
        load_state_dict_flexible(net_model, checkpoint["net_model"], FLAGS.parallel)
        load_state_dict_flexible(ema_model, checkpoint["ema_model"], FLAGS.parallel)
        optim.load_state_dict(checkpoint["optim"])
        sched.load_state_dict(checkpoint["sched"])
        start_step = checkpoint.get("step", 0)
        # Load class centroids if available
        if "class_centroids" in checkpoint:
            class_centroids = checkpoint["class_centroids"]
            print("Loaded class centroids from checkpoint")
        print(f"Resumed from step {start_step}")

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + f"_anchor_{FLAGS.anchor_loss_type}/"
    os.makedirs(savedir, exist_ok=True)

    with trange(start_step, FLAGS.total_steps, dynamic_ncols=True, initial=start_step, total=FLAGS.total_steps) as pbar:
        for step in pbar:
            # Periodically update centroids
            if step > 0 and step % FLAGS.centroid_update_freq == 0:
                class_centroids = compute_class_centroids(dataset, class_sampler, device=device)
            
            optim.zero_grad()
            
            # Sample a batch from the same class D_i
            x1_batch, sampled_class = next(dataloader_iter)
            x1 = x1_batch.to(device)
            
            # Get the centroid e(x_1) for this class
            e_x1 = class_centroids[sampled_class].unsqueeze(0).expand_as(x1)
            
            # Standard flow matching setup
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            
            # Standard flow matching loss: E[||v_t - u_t||^2]
            loss_fm = torch.mean((vt - ut) ** 2)
            
            # Anchor regularization term
            # Compute e(x_1)_t with same noise and time as x_t
            batch_size = x1.shape[0]
            
            # Sample same time t for both x_1 and e(x_1)
            t_anchor = t  # Use the same time
            eps = torch.randn_like(x0)
            
            # Compute x_t and e(x_1)_t with the same noise and time
            mu_x_t = t_anchor.reshape(-1, 1, 1, 1) * x1 + (1 - t_anchor.reshape(-1, 1, 1, 1)) * x0
            mu_e_t = t_anchor.reshape(-1, 1, 1, 1) * e_x1 + (1 - t_anchor.reshape(-1, 1, 1, 1)) * x0
            
            # Add same noise to both (if sigma > 0)
            sigma_val = FM.sigma
            x_t = mu_x_t + sigma_val * eps
            e_t = mu_e_t + sigma_val * eps
            
            # Compute v(x_t, t) and v(e(x_1)_t, t)
            v_xt = net_model(t_anchor, x_t)
            v_et = net_model(t_anchor, e_t)
            
            # Choose loss type
            if FLAGS.anchor_loss_type == "full":
                # Eq. 5: lambda * ||v(x_t, t) - v(e(x_1)_t, t) + e(x_1) - x_1||^2
                loss_anchor = torch.mean((v_xt - v_et + e_x1 - x1) ** 2)
            elif FLAGS.anchor_loss_type == "simple":
                # Eq. 6: lambda * ||v(x_t, t) - v(e(x_1)_t, t)||^2
                loss_anchor = torch.mean((v_xt - v_et) ** 2)
            else:
                raise ValueError(f"Unknown anchor_loss_type: {FLAGS.anchor_loss_type}")
            
            # Combined loss: L_anchor = L_fm + lambda * L_anchor_reg
            loss = loss_fm + FLAGS.lambda_anchor * loss_anchor
            loss.backward()
            
            # Compute gradient norm before clipping
            total_norm = 0.0
            for p in net_model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)
            
            # Compute vector field value mean norm
            with torch.no_grad():
                vt_norm = torch.mean(torch.norm(vt.reshape(vt.shape[0], -1), dim=1)).item()
            
            # Log metrics to wandb every 100 steps
            if step % 100 == 0:
                wandb.log({
                    "loss_fm": loss_fm.item(),
                    "anchor_loss": loss_anchor.item(),
                    "grad_norm": grad_norm,
                    "vector_field_mean_norm": vt_norm,
                    "step": step, 
                    "lr": sched.get_last_lr()[0],
                }, step=step)
            
            pbar.set_postfix({
                "loss": loss.item(), 
                "loss_fm": loss_fm.item(), 
                "loss_anchor": loss_anchor.item(), 
                "class": sampled_class
            })

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0 and step > 0:
                generate_samples(net_model, FLAGS.parallel, savedir, start_step + step, net_="normal")
                generate_samples(ema_model, FLAGS.parallel, savedir, start_step + step, net_="ema")
                
                # Compute and log FID score
                print("Computing FID score...")
                fid_score = compute_fid_score(ema_model, FLAGS.parallel, num_gen=10000)
                if fid_score is not None:
                    print(f"FID Score at step {step}: {fid_score}")
                    wandb.log({"fid": fid_score, "step": step}, step=step)
                
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                        "class_centroids": class_centroids,
                    },
                    savedir + f"{FLAGS.model}_cifar10_anchor_{FLAGS.lambda_name}_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
