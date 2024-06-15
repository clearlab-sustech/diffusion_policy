import torch
from einops import rearrange


def proximate_mu(alpha_T_bar, alpha_N_bar, noised_x, original_x):
    alpha_T2N_bar = alpha_N_bar / alpha_T_bar
    coef_1 = alpha_T2N_bar**0.5 * (1 - alpha_T_bar) / (1 - alpha_N_bar)
    coef_2 = alpha_T_bar**0.5 * (1 - alpha_T2N_bar) / (1 - alpha_N_bar)

    return coef_1 * noised_x + coef_2 * original_x


def beta_bar(alpha_T_bar, alpha_N_bar):
    alpha_T2N_bar = alpha_N_bar / alpha_T_bar
    return (1 - alpha_T2N_bar) * (1 - alpha_T_bar) / (1 - alpha_N_bar)


def estimate_noise(x_t, x_0, alpha_cumprod, timesteps):
    alpha_t_bat = torch.tensor([alpha_cumprod[ts] for ts in timesteps]).to(x_t.device)
    coef_1 = 1 / (1 - alpha_t_bat) ** 0.5
    coef_2 = -(alpha_t_bat**0.5) / (1 - alpha_t_bat) ** 0.5

    coef_1 = rearrange(coef_1, "bs -> bs 1 1")
    coef_2 = rearrange(coef_2, "bs -> bs 1 1")

    return coef_1 * x_t + coef_2 * x_0
