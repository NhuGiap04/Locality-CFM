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
from utils_cifar import ema, generate_samples, infiniteloop

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

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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


class ClassConditionedSampler:
    """
    A sampler that provides batches of samples from the same class.
    
    This is useful for local-aware training where we want to ensure
    all samples in a batch come from the same class.
    """
    
    def __init__(self, dataset, batch_size, num_classes=10):
        """
        Args:
            dataset: PyTorch dataset with (image, label) pairs
            batch_size: Number of samples per batch
            num_classes: Number of classes in the dataset (default: 10 for CIFAR-10)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Organize dataset indices by class
        self.class_to_indices = {i: [] for i in range(num_classes)}
        for idx, (_, label) in enumerate(dataset):
            self.class_to_indices[label].append(idx)
        
        # Convert to tensors for efficient indexing
        for class_id in range(num_classes):
            self.class_to_indices[class_id] = torch.tensor(self.class_to_indices[class_id])
        
        # Store the current position in each class
        self.class_positions = {i: 0 for i in range(num_classes)}
        
        # Shuffle indices for each class
        for class_id in range(num_classes):
            perm = torch.randperm(len(self.class_to_indices[class_id]))
            self.class_to_indices[class_id] = self.class_to_indices[class_id][perm]
    
    def sample_batch_from_class(self, class_id=None):
        """
        Sample a batch of images all from the same class.
        
        Args:
            class_id: Specific class to sample from. If None, randomly select a class.
        
        Returns:
            batch: Tensor of shape (batch_size, C, H, W)
            class_id: The class that was sampled
        """
        # Randomly select a class if not specified
        if class_id is None:
            class_id = torch.randint(0, self.num_classes, (1,)).item()
        
        # Get indices for this class
        class_indices = self.class_to_indices[class_id]
        current_pos = self.class_positions[class_id]
        
        # If we don't have enough samples left in this class, reshuffle
        if current_pos + self.batch_size > len(class_indices):
            perm = torch.randperm(len(class_indices))
            self.class_to_indices[class_id] = class_indices[perm]
            current_pos = 0
            self.class_positions[class_id] = 0
        
        # Get batch indices
        batch_indices = class_indices[current_pos:current_pos + self.batch_size]
        self.class_positions[class_id] = current_pos + self.batch_size
        
        # Load the actual data
        batch_data = []
        for idx in batch_indices:
            img, _ = self.dataset[idx.item()]
            batch_data.append(img)
        
        batch = torch.stack(batch_data)
        
        return batch, class_id
    
    def __iter__(self):
        """Iterator that yields batches indefinitely."""
        while True:
            batch, class_id = self.sample_batch_from_class()
            yield batch, class_id


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
    class_sampler = ClassConditionedSampler(dataset, FLAGS.batch_size, num_classes=10)
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

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
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
            
            # Local regularization term: lambda * E[||v(u_t, t) - v(s_t, t) + s - u||^2]
            # Sample pairs of points (u, s) from the same batch (same class)
            batch_size = x1.shape[0]
            if batch_size > 1:
                # Randomly pair samples within the batch
                perm = torch.randperm(batch_size, device=device)
                u = x1  # First set of samples
                s = x1[perm]  # Permuted samples (different pairing)
                
                # Use same noise x0 for both, but different endpoints u and s
                # Sample the same time t for both u and s
                t_reg = torch.rand(batch_size).type_as(x1)
                
                # Compute u_t and s_t with the same noise and time
                eps = torch.randn_like(x0)
                mu_u_t = t_reg.reshape(-1, 1, 1, 1) * u + (1 - t_reg.reshape(-1, 1, 1, 1)) * x0
                mu_s_t = t_reg.reshape(-1, 1, 1, 1) * s + (1 - t_reg.reshape(-1, 1, 1, 1)) * x0
                
                # Add same noise to both (if sigma > 0)
                sigma = FM.sigma
                u_t = mu_u_t + sigma * eps
                s_t = mu_s_t + sigma * eps
                
                # Compute v(u_t, t) and v(s_t, t)
                v_ut = net_model(t_reg, u_t)
                v_st = net_model(t_reg, s_t)
                
                # Regularization: ||v(u_t, t) - v(s_t, t) + s - u||^2
                loss_local = torch.mean((v_ut - v_st + s - u) ** 2)
            else:
                loss_local = torch.tensor(0.0, device=device)
            
            # Combined loss: L_local = L_fm + lambda * L_local_reg
            loss = loss_fm + FLAGS.lambda_local * loss_local
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)
            
            # Log loss to wandb with class information
            wandb.log({
                "loss_fm": loss_fm.item(),
                "regularized_loss": loss_local.item() if batch_size > 1 else 0.0,
                "step": step, 
                "lr": sched.get_last_lr()[0],
            })
            pbar.set_postfix({"loss": loss.item(), "loss_fm": loss_fm.item(), "loss_local": loss_local.item() if batch_size > 1 else 0.0, "class": sampled_class})

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                
                # Compute and log FID score
                print("Computing FID score...")
                fid_score = compute_fid_score(ema_model, FLAGS.parallel, num_gen=10000)
                if fid_score is not None:
                    print(f"FID Score at step {step}: {fid_score}")
                    wandb.log({"fid": fid_score, "step": step})
                
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
