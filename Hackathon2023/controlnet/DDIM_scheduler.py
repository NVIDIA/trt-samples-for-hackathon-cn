import numpy as np
import torch
from ldm.modules.diffusionmodules.util import (make_beta_schedule,
                                               make_ddim_sampling_parameters,
                                               make_ddim_timesteps)


class DDIM():

    def __init__(self, *args, **kwargs):
        self.ddpm_num_timesteps = 1000
        self.schedule = "linear"
        self.ddim_timesteps = None
        self.ddim_sigmas = None
        self.ddim_alphas = None
        self.ddim_alphas_prev = None
        self.ddim_sqrt_one_minus_alphas = None
        self.ddim_sigmas_for_original_num_steps = None
        betas = make_beta_schedule(schedule="linear",
                                   n_timestep=1000,
                                   linear_start=0.00085,
                                   linear_end=0.012,
                                   cosine_s=0.008)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0, dtype=np.float32)
        alphas_cumprod_prev = np.append(1.,
                                        alphas_cumprod[:-1]).astype(np.float32)

        self.betas = torch.from_numpy(betas.astype(np.float32))
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod)
        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev)

        self.sqrt_alphas_cumprod = torch.from_numpy(
            np.sqrt(alphas_cumprod).astype(np.float32))
        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(
            np.sqrt(1. - alphas_cumprod).astype(np.float32))
        self.log_one_minus_alphas_cumprod = torch.from_numpy(
            np.log(1. - alphas_cumprod).astype(np.float32))
        self.sqrt_recip_alphas_cumprod = torch.from_numpy(
            np.sqrt(1. / alphas_cumprod).astype(np.float32))
        self.sqrt_recipm1_alphas_cumprod = torch.from_numpy(
            np.sqrt(1. / alphas_cumprod - 1).astype(np.float32))

        self.make_schedule(ddim_num_steps=20, ddim_eta=0.0, verbose=False)

    def make_schedule(self,
                      ddim_num_steps,
                      ddim_discretize="uniform",
                      ddim_eta=0.,
                      verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose)
        alphas_cumprod = self.alphas_cumprod.cpu()
        assert alphas_cumprod.shape[
            0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod,
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose)

        self.ddim_sigmas = ddim_sigmas.to("cuda")
        self.ddim_alphas = ddim_alphas.to("cuda")
        self.ddim_alphas_prev = torch.from_numpy(ddim_alphas_prev).to("cuda")
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. -
                                                     ddim_alphas).to("cuda")
        self.ddim_sigmas_for_original_num_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) *
            (1 - self.alphas_cumprod / self.alphas_cumprod_prev))

        self.ddim_timesteps = np.flip(self.ddim_timesteps)


@torch.jit.script
def step(x: torch.Tensor, latent: torch.Tensor, index: int,
         ddim_alphas: torch.Tensor, ddim_alphas_prev: torch.Tensor,
         ddim_sqrt_one_minus_alphas: torch.Tensor) -> torch.Tensor:
    model_t = latent[0:1, :, :, :]
    model_uncond = latent[1:, :, :, :]
    e_t = model_uncond + 9. * (model_t - model_uncond)

    a_t = ddim_alphas[index]
    a_prev = ddim_alphas_prev[index]
    sqrt_one_minus_at = ddim_sqrt_one_minus_alphas[index]
    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    dir_xt = (1. - a_prev).sqrt() * e_t
    new_x = a_prev.sqrt() * pred_x0 + dir_xt

    return new_x
