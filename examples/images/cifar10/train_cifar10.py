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

# Checkpoint
flags.DEFINE_string("checkpoint_path", "", help="path to checkpoint file to resume training from")

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


def train(argv):
    # Initialize wandb
    wandb.init(
        project="conditional-flow-matching",
        name=f"{FLAGS.model}_cifar10",
        config={
            "model": FLAGS.model,
            "lr": FLAGS.lr,
            "total_steps": FLAGS.total_steps,
            "batch_size": FLAGS.batch_size,
            "ema_decay": FLAGS.ema_decay,
            "num_channel": FLAGS.num_channel,
            "grad_clip": FLAGS.grad_clip,
            "warmup": FLAGS.warmup,
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
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

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
        if FLAGS.parallel:
            net_model.module.load_state_dict(checkpoint["net_model"])
            ema_model.module.load_state_dict(checkpoint["ema_model"])
        else:
            net_model.load_state_dict(checkpoint["net_model"])
            ema_model.load_state_dict(checkpoint["ema_model"])
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

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    with trange(start_step, FLAGS.total_steps, dynamic_ncols=True, initial=start_step, total=FLAGS.total_steps) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new
            
            # Log loss to wandb
            wandb.log({"loss_fm": loss.item(), "step": step, "lr": sched.get_last_lr()[0]})
            pbar.set_postfix({"loss": loss.item()})

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
                    savedir + f"{FLAGS.model}_cifar10_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
