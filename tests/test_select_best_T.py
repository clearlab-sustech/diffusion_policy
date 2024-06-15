import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def kl_divergence(mu, var):
    # 计算 mu 的 L2 范数
    mu_norm = torch.sum(mu**2, dim=[1, 2])

    # 动作的维度
    feature_dim = mu.size(2)

    # 确保 alpha 是一个张量
    var_tensor = (
        torch.tensor(var, dtype=mu.dtype, device=mu.device)
        if isinstance(var, float)
        else var.clone().detach()
    )

    # 计算 KL 散度
    kl_divergence = 0.5 * (
        feature_dim * var_tensor
        + mu_norm
        - feature_dim
        - feature_dim * torch.log(var_tensor)
    )

    return kl_divergence


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.parent.joinpath("diffusion_policy", "config")
    ),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    train_dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(dataset.get_normalizer().state_dict())
    point = []
    max_v = 21
    for ts in range(1, max_v):
        scheduler = DDPMScheduler(
            num_train_timesteps=ts,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",  # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
            clip_sample=True,  # required when predict_epsilon=False
            prediction_type="epsilon",  # or sample
            max_beta=0.5
        )

        kl = list()
        idx = 0
        for batch in train_dataloader:
            if idx >= 100:
                break
            naction = normalizer["action"].normalize(batch["action"])
            bs, T, dim = naction.shape
            I = torch.full_like(naction, 1.0)

            timesteps = torch.full(
                (bs,),
                ts - 1,
                dtype=torch.int64,
            ).long()
            scheduler.add_noise(naction, torch.randn_like(naction), timesteps)

            alpha_t = scheduler.alphas_cumprod[-1]

            mean_1 = alpha_t**0.5 * naction
            kl.append(kl_divergence(mean_1, 1 - alpha_t).mean().detach().item())
            idx += 1
        print(
            f"timestep: {ts}, kl: {np.mean(kl)}, alpha_t: {scheduler.alphas}, alpha_t_bar: {scheduler.alphas_cumprod}"
        )
        point.append(np.mean(kl))
    point_ts = [idx for idx in range(1, max_v)]

    plt.plot(point_ts, point)
    plt.xlabel("timestep")
    plt.ylabel("kl_divergence")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
