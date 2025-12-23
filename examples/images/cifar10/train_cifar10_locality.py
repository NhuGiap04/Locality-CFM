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
from utils_cifar import ema, generate_samples, ClassConditionedSampler, compute_local_loss, visualize_clusters

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
flags.DEFINE_float("lambda_local", 1.0, help="Lambda value as string for logging")
flags.DEFINE_string("lambda_name", "1e0", help="size of local region for local regularization")
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


def train(argv):
    # Initialize wandb
    wandb.init(
        project="conditional-flow-matching",
        name=f"{FLAGS.model}_cifar10_local_{FLAGS.lambda_name}",
        config={
            "model": FLAGS.model,
            "lr": FLAGS.lr,
            "total_steps": FLAGS.total_steps,
            "batch_size": FLAGS.batch_size,
            "ema_decay": FLAGS.ema_decay,
            "num_channel": FLAGS.num_channel,
            "grad_clip": FLAGS.grad_clip,
            "warmup": FLAGS.warmup,
            "lambda_local": FLAGS.lambda_local,
            "lambda_name": FLAGS.lambda_name,
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
    
    # Use the class-conditioned sampler instead of regular dataloader
    class_sampler = ClassConditionedSampler(dataset, FLAGS.batch_size, num_classes=10, num_subclasses=FLAGS.num_subclasses)

    if FLAGS.visualize:
        print("Visualizing class clusters...")
        visualize_clusters(class_sampler, dataset)

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
    ).to(device)  # new dropout + bs of 128

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

    savedir = FLAGS.output_dir + FLAGS.model + f"_local_{FLAGS.lambda_name}/"
    os.makedirs(savedir, exist_ok=True)

    with trange(start_step, FLAGS.total_steps, dynamic_ncols=True, initial=start_step, total=FLAGS.total_steps) as pbar:
        for step in pbar:
            optim.zero_grad()
            
            # Sample a batch from the same class
            x1_batch, sampled_class = next(dataloader_iter)
            x1 = x1_batch.to(device)
            
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            
            # Standard flow matching loss: E[||v_t - u_t||^2]
            loss_fm = torch.mean((vt - ut) ** 2)
            
            # Local regularization term using shared function
            # Samples are from the same class due to ClassConditionedSampler
            loss_local = compute_local_loss(net_model, x1, x0, sigma=FM.sigma)
            
            # Combined loss: L_local = L_fm + lambda * L_local_reg
            loss = loss_fm + FLAGS.lambda_local * loss_local
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
                    "local_loss": loss_local.item(),
                    "grad_norm": grad_norm,
                    "vector_field_mean_norm": vt_norm,
                    "step": step, 
                    "lr": sched.get_last_lr()[0],
                }, step=step)
            
            pbar.set_postfix({
                "loss": loss.item(), 
                "loss_fm": loss_fm.item(), 
                "loss_local": loss_local.item(), 
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
                    },
                    savedir + f"{FLAGS.model}_cifar10_local_{FLAGS.lambda_name}_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
