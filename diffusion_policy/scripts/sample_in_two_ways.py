"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import tqdm
import json
import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange, reduce
from omegaconf import OmegaConf
import pathlib
import datetime


ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace



@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-i", "--iterations", type=int, required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, output_dir, iterations, device):
    if os.path.exists(output_dir):
        click.confirm(
            f"Output path {output_dir} already exists! Overwrite?", abort=True
        )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    wandb_run = wandb.init(
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.logging,
    )
    wandb.config.update(
        {
            "output_dir": output_dir,
        }
    )

    # loading dataset
    cfg.dataloader.shuffle = False
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    train_dataloader = DataLoader(dataset, **cfg.dataloader)

    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    At2A_ls = list()
    Ep2A_ls = list()
    global_step = 0
    for iter in range(iterations):
        with torch.no_grad():
            with tqdm.tqdm(
                val_dataloader,
                desc=f"Testing: ",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for _, batch in enumerate(tepoch):
                    batch = dict_apply(
                        batch,
                        lambda x: x.to(device, non_blocking=True),
                    )

                    # add noise to actions
                    traj = policy.normalizer["action"].normalize(batch["action"])
                    noise = torch.randn(traj.shape, device=device)
                    timesteps = torch.full(
                        (traj.shape[0],),
                        policy.noise_scheduler.config.num_train_timesteps - 1,
                        dtype=torch.int64,
                        device=device,
                    ).long()
                    noise_traj = policy.noise_scheduler.add_noise(
                        traj, noise, timesteps
                    )
                    At2A = policy.predict_action(batch["obs"], noise_traj)[
                        "action_pred"
                    ]
                    Ep2A = policy.predict_action(batch["obs"])["action_pred"]

                    At2A_mse = reduce(
                        F.mse_loss(At2A, traj, reduction="none"),
                        "b ... -> b (...)",
                        "mean",
                    ).mean()
                    Ep2A_mse = reduce(
                        F.mse_loss(Ep2A, traj, reduction="none"),
                        "b ... -> b (...)",
                        "mean",
                    ).mean()
                    At2A_ls.append(At2A_mse)
                    Ep2A_ls.append(Ep2A_mse)
                    step_log = {
                        "At2A_mse_batch": At2A_mse,
                        "Ep2A_mse_batch": Ep2A_mse,
                    }
                    wandb_run.log(step_log, step=global_step)
                    global_step += 1

                At2A_mse_error = torch.mean(torch.tensor(At2A_ls)).item()
                Ep2A_mse_error = torch.mean(torch.tensor(Ep2A_ls)).item()
                At2A_mse_error_max = torch.max(torch.tensor(At2A_ls)).item()
                Ep2A_mse_error_max = torch.max(torch.tensor(Ep2A_ls)).item()
                iter_log = {
                    "At2A_mse_avg": At2A_mse_error,
                    "02A_mse_avg": Ep2A_mse_error,
                    "At2A_mse_max": At2A_mse_error_max,
                    "02A_mse_max": Ep2A_mse_error_max,
                }
                wandb_run.log(iter_log, step=global_step)


if __name__ == "__main__":
    main()
