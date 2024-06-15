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
import numpy as np
from torch.utils.data import DataLoader
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def create_or_append_hdf5(file_path, noise, pred_action):
    import h5py

    with h5py.File(file_path, "w") as file:
        data_group = file.create_group("data")

        data_group.create_dataset("noise", data=noise)
        data_group.create_dataset("pred_action", data=pred_action)


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-dp", "--dataset_path", required=True)
@click.option("-n", "--num", type=int, default=1)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, output_dir, dataset_path, num, device):
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

    # loading dataset
    cfg.dataloader.shuffle = False  # turn off shuffle mode
    cfg.task.dataset.val_ratio = 0.0  # using all data as training data
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    train_dataloader = DataLoader(dataset, **cfg.dataloader)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    pred_actions = list()
    epsilons = list()
    with torch.no_grad():
        with tqdm.tqdm(
            train_dataloader,
            desc=f"Generate new dataset: ",
            leave=False,
            mininterval=cfg.training.tqdm_interval_sec,
        ) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(
                    batch,
                    lambda x: x.to(device, non_blocking=True),
                )
                pred_action_set = list()
                epsilon_set = list()
                for _ in range(num):
                    opt = policy.predict_action(batch["obs"])
                    pred_action_set.append(opt["action_pred"].cpu().numpy())
                    epsilon_set.append(opt["epsilon"].cpu().numpy())

                pred_action_set = np.stack(
                    pred_action_set, axis=1
                )  # (batch_size, num, seq, action_dim)
                epsilon_set = np.stack(
                    epsilon_set, axis=1
                )  # batch_size, num, horizon, action_dim)

                pred_actions.append(pred_action_set)
                epsilons.append(epsilon_set)

            create_or_append_hdf5(
                dataset_path,
                np.concatenate(epsilons, axis=0),
                np.concatenate(pred_actions, axis=0),
            )

            print(f"New dataset is saved at {dataset_path}")


if __name__ == "__main__":
    main()
