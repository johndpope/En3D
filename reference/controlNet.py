
"""
A Canny detection algorithm for RGB images.

Taken from GeeksforGeeks: 
https://www.geeksforgeeks.org/implement-canny-edge-detector-in-python-using-opencv/
"""

import numpy as np 
import cv2 
from torchvision import transforms


class AddCannyImage:
    """
    Wrapper for Canny edge detection function to be used
    as a transform in PyTorch's torchvision.transforms.Compose
    """
    def __init__(self, threshold=100):
        self.threshold = threshold

    def __call__(self, img):
        # Convert image to grayscale
        gray = transforms.Grayscale()(img)
        gray.squeeze_(0)

        # Convert grayscale image to numpy array
        gray = np.array(gray)

        # Apply Canny edge detection
        canny = Canny_detector(gray)

        # Convert Canny image to tensor
        canny = transforms.ToTensor()(canny)

        # Change to half precision
        canny = canny.half()
        
        return canny
	
def Canny_detector(img, weak_th = None, strong_th = None): 
	
	# Noise reduction step 
	img = cv2.GaussianBlur(img, (5, 5), 1.4) 
	
	# Calculating the gradients 
	gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3) 
	gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3) 
	
	# Conversion of Cartesian coordinates to polar 
	mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True) 
	
	# setting the minimum and maximum thresholds 
	# for double thresholding 
	mag_max = np.max(mag) 
	if not weak_th:weak_th = mag_max * 0.1
	if not strong_th:strong_th = mag_max * 0.5
	
	# getting the dimensions of the input image 
	height, width = img.shape 
	
	# Looping through every pixel of the grayscale 
	# image 
	for i_x in range(width): 
		for i_y in range(height): 
			
			grad_ang = ang[i_y, i_x] 
			grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang) 
			
			# selecting the neighbours of the target pixel 
			# according to the gradient direction 
			# In the x axis direction 
			if grad_ang<= 22.5: 
				neighb_1_x, neighb_1_y = i_x-1, i_y 
				neighb_2_x, neighb_2_y = i_x + 1, i_y 
			
			# top right (diagonal-1) direction 
			elif grad_ang>22.5 and grad_ang<=(22.5 + 45): 
				neighb_1_x, neighb_1_y = i_x-1, i_y-1
				neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
			
			# In y-axis direction 
			elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90): 
				neighb_1_x, neighb_1_y = i_x, i_y-1
				neighb_2_x, neighb_2_y = i_x, i_y + 1
			
			# top left (diagonal-2) direction 
			elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135): 
				neighb_1_x, neighb_1_y = i_x-1, i_y + 1
				neighb_2_x, neighb_2_y = i_x + 1, i_y-1
			
			# Now it restarts the cycle 
			elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180): 
				neighb_1_x, neighb_1_y = i_x-1, i_y 
				neighb_2_x, neighb_2_y = i_x + 1, i_y 
			
			# Non-maximum suppression step 
			if width>neighb_1_x>= 0 and height>neighb_1_y>= 0: 
				if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]: 
					mag[i_y, i_x]= 0
					continue

			if width>neighb_2_x>= 0 and height>neighb_2_y>= 0: 
				if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]: 
					mag[i_y, i_x]= 0

	weak_ids = np.zeros_like(img) 
	strong_ids = np.zeros_like(img)			 
	ids = np.zeros_like(img) 
	
	# double thresholding step 
	for i_x in range(width): 
		for i_y in range(height): 
			
			grad_mag = mag[i_y, i_x] 
			
			if grad_mag<weak_th: 
				mag[i_y, i_x]= 0
			elif strong_th>grad_mag>= weak_th: 
				ids[i_y, i_x]= 1
			else: 
				ids[i_y, i_x]= 2
	
	
	# finally returning the magnitude of 
	# gradients of edges 
	return mag 
"""
This file contains an implementation of the paper 
"Adding Conditional Control to Text-to-Image Diffusion Models" 
by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

Reference:
Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. (2023). 
Adding Conditional Control to Text-to-Image Diffusion Models. 
Retrieved from https://arxiv.org/abs/2302.05543
"""

import torch
import torch.nn as nn
from torch.special import expm1
import math
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from pathlib import Path
import os
from tqdm import tqdm
from ema_pytorch import EMA
import matplotlib.pyplot as plt

# helper
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

class ControlNet(nn.Module):
    def __init__(
        self, 
        unet,
        controlnet,
        image_size,
        noise_size=64,
        pred_param='v', 
        schedule='shifted_cosine', 
        steps=512
    ):
        super().__init__()

        # Training objective
        assert pred_param in ['v', 'eps'], "Invalid prediction parameterization. Must be 'v' or 'eps'"
        self.pred_param = pred_param

        # Sampling schedule
        assert schedule in ['cosine', 'shifted_cosine'], "Invalid schedule. Must be 'cosine' or 'shifted_cosine'"
        self.schedule = schedule
        self.noise_d = noise_size
        self.image_d = image_size

        # Model
        assert isinstance(unet, nn.Module), "Model must be an instance of torch.nn.Module."
        self.model = unet
        self.controlnet = controlnet

        # Lock the parameters of the UNet
        self.model.requires_grad_(False)

        num_params = sum(p.numel() for p in self.model.parameters())
        num_params += sum(p.numel() for p in self.controlnet.parameters())
        print(f"Number of parameters: {num_params}")

        # Steps
        self.steps = steps

    def diffuse(self, x, alpha_t, sigma_t):
        """
        Function to diffuse the input tensor x to a timepoint t with the given alpha_t and sigma_t.

        Args:
        x (torch.Tensor): The input tensor to diffuse.
        alpha_t (torch.Tensor): The alpha value at timepoint t.
        sigma_t (torch.Tensor): The sigma value at timepoint t.

        Returns:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        eps_t (torch.Tensor): The noise tensor at timepoint t.
        """
        eps_t = torch.randn_like(x)

        z_t = alpha_t * x + sigma_t * eps_t

        return z_t, eps_t

    def logsnr_schedule_cosine(self, t, logsnr_min=-15, logsnr_max=15):
        """
        Function to compute the logSNR schedule at timepoint t with cosine:

        logSNR(t) = -2 * log (tan (pi * t / 2))

        Taking into account boundary effects, the logSNR value at timepoint t is computed as:

        logsnr_t = -2 * log(tan(t_min + t * (t_max - t_min)))

        Args:
        t (int): The timepoint t.
        logsnr_min (int): The minimum logSNR value.
        logsnr_max (int): The maximum logSNR value.

        Returns:
        logsnr_t (float): The logSNR value at timepoint t.
        """
        logsnr_max = logsnr_max + math.log(self.noise_d / self.image_d)
        logsnr_min = logsnr_min + math.log(self.noise_d / self.image_d)
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))

        logsnr_t = -2 * log(torch.tan(torch.tensor(t_min + t * (t_max - t_min))))

        return logsnr_t

    def logsnr_schedule_cosine_shifted(self, t):
        """
        Function to compute the logSNR schedule at timepoint t with shifted cosine:

        logSNR_shifted(t) = logSNR(t) + 2 * log(noise_d / image_d)

        Args:
        t (int): The timepoint t.
        image_d (int): The image dimension.
        noise_d (int): The noise dimension.

        Returns:
        logsnr_t_shifted (float): The logSNR value at timepoint t.
        """
        logsnr_t = self.logsnr_schedule_cosine(t)
        logsnr_t_shifted = logsnr_t + 2 * math.log(self.noise_d / self.image_d)

        return logsnr_t_shifted
        
    def clip(self, x):
        """
        Function to clip the input tensor x to the range [-1, 1].

        Args:
        x (torch.Tensor): The input tensor to clip.

        Returns:
        x (torch.Tensor): The clipped tensor.
        """
        return torch.clamp(x, -1, 1)

    @torch.no_grad()
    def ddpm_sampler_step(self, z_t, pred, logsnr_t, logsnr_s):
        """
        Function to perform a single step of the DDPM sampler.

        Args:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        pred (torch.Tensor): The predicted value from the model (v or eps).
        logsnr_t (float): The logSNR value at timepoint t.
        logsnr_s (float): The logSNR value at the sampling timepoint s.

        Returns:
        z_s (torch.Tensor): The diffused tensor at sampling timepoint s.
        """
        c = -expm1(logsnr_t - logsnr_s)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        if self.pred_param == 'v':
            x_pred = alpha_t * z_t - sigma_t * pred
        elif self.pred_param == 'eps':
            x_pred = (z_t - sigma_t * pred) / alpha_t

        x_pred = self.clip(x_pred)

        mu = alpha_s * (z_t * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s ** 2) * c

        return mu, variance

    @torch.no_grad()
    def sample(self, x, c):
        """
        Standard DDPM sampling procedure. Begun by sampling z_T ~ N(0, 1)
        and then repeatedly sampling z_s ~ p(z_s | z_t)

        Args:
        x (torch.Tensor): An input tensor that is of the desired shape.
        c (torch.Tensor): The control tensor.

        Returns:
        x_pred (torch.Tensor): The predicted tensor.
        """
        z_t = torch.randn(x.shape).to(x.device)
        c = c.to(x.device)

        # Steps T -> 1
        for t in reversed(range(1, self.steps+1)):
            u_t = t / self.steps
            u_s = (t - 1) / self.steps

            if self.schedule == 'cosine':
                logsnr_t = self.logsnr_schedule_cosine(u_t)
                logsnr_s = self.logsnr_schedule_cosine(u_s)
            elif self.schedule == 'shifted_cosine':
                logsnr_t = self.logsnr_schedule_cosine_shifted(u_t)
                logsnr_s = self.logsnr_schedule_cosine_shifted(u_s)

            logsnr_t = logsnr_t.to(x.device)
            logsnr_s = logsnr_s.to(x.device)

            down_residuals, mid_residuals = self.controlnet(z_t, logsnr_t, c)
            pred = self.model(
                z_t, 
                logsnr_t,
                downblock_additional_residuals=down_residuals,
                midblock_additional_residuals=mid_residuals
                )
            mu, variance = self.ddpm_sampler_step(z_t, pred, torch.tensor(logsnr_t), torch.tensor(logsnr_s))
            z_t = mu + torch.randn_like(mu) * torch.sqrt(variance)

        # Final step
        if self.schedule == 'cosine':
            logsnr_1 = self.logsnr_schedule_cosine(1/self.steps)
            logsnr_0 = self.logsnr_schedule_cosine(0)
        elif self.schedule == 'shifted_cosine':
            logsnr_1 = self.logsnr_schedule_cosine_shifted(1/self.steps)
            logsnr_0 = self.logsnr_schedule_cosine_shifted(0)

        logsnr_1 = logsnr_1.to(x.device)
        logsnr_0 = logsnr_0.to(x.device)

        down_residuals, mid_residuals = self.controlnet(z_t, logsnr_1, c)
        pred = self.model(
            z_t, 
            logsnr_1,
            downblock_additional_residuals=down_residuals,
            midblock_additional_residuals=mid_residuals
            )
        x_pred, _ = self.ddpm_sampler_step(z_t, pred, torch.tensor(logsnr_1), torch.tensor(logsnr_0))
        
        x_pred = self.clip(x_pred)

        # Convert x_pred to the range [0, 1]
        x_pred = (x_pred + 1) / 2

        return x_pred
    
    def loss(self, x, c):
        """
        A function to compute the loss of the model. The loss is computed as the mean squared error
        between the predicted noise tensor and the true noise tensor. Various prediction parameterizations
        imply various weighting schemes as outlined in Kingma et al. (2023)

        Args:
        x (torch.Tensor): The input tensor.
        c (torch.Tensor): The control tensor.

        Returns:
        loss (torch.Tensor): The loss value.
        """
        t = torch.rand(x.shape[0])
        c = c.to(x.device)

        if self.schedule == 'cosine':
            logsnr_t = self.logsnr_schedule_cosine(t)
        elif self.schedule == 'shifted_cosine':
            logsnr_t = self.logsnr_schedule_cosine_shifted(t)

        logsnr_t = logsnr_t.to(x.device)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        z_t, eps_t = self.diffuse(x, alpha_t, sigma_t)
        down_residuals, mid_residuals = self.controlnet(z_t, logsnr_t, c)
        pred = self.model(
            z_t, 
            logsnr_t,
            downblock_additional_residuals=down_residuals,
            midblock_additional_residuals=mid_residuals
            )

        if self.pred_param == 'v':
            eps_pred = sigma_t * z_t + alpha_t * pred
        else: 
            eps_pred = pred

        # Apply min-SNR weighting (https://arxiv.org/pdf/2303.09556)
        snr = torch.exp(logsnr_t).clamp_(max = 5)
        if self.pred_param == 'v':
            weight = 1 / (1 + snr)
        else:
            weight = 1 / snr

        weight = weight.view(-1, 1, 1, 1)

        loss = torch.mean(weight * (eps_pred - eps_t) ** 2)

        return loss

    def train_loop(self, config, optimizer, train_dataloader, lr_scheduler):
        """
        A function to train the model.

        Args:
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        """
        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_dir=os.path.join(config.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            if config.push_to_hub:
                repo_id = create_repo(
                    repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
                ).repo_id
                print(repo_id)

        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare( 
            self.controlnet, optimizer, train_dataloader, lr_scheduler
        )
        self.model.to(accelerator.device)

        # Create an EMA model
        ema = EMA(
            controlnet,
            beta=0.9999,
            update_after_step=100,
            update_every=10
        )

        global_step = 0

        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                x = batch["images"]
                c = batch["conditions"]

                with accelerator.accumulate(controlnet):
                    loss = self.loss(x, c)
                    loss = loss.to(next(controlnet.parameters()).dtype)
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Update EMA model parameters
                ema.update()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images
            if accelerator.is_main_process:
                self.controlnet = accelerator.unwrap_model(controlnet)

                # Make directory for saving images
                os.makedirs(os.path.join(config.output_dir, "images"), exist_ok=True)

                if epoch % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    sample = self.sample(x[0].unsqueeze(0), c[0].unsqueeze(0))
                    sample = sample.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    image_path = os.path.join(config.output_dir, "images", f"sample_{epoch}.png")

                    # Create a side-by-side comparison of c[0] and the generated image
                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(c[0].squeeze().cpu().numpy())
                    ax[0].set_title("Control Tensor")
                    ax[0].axis("off")
                    ax[1].imshow(sample[0])
                    ax[1].set_title("Generated Image")
                    ax[1].axis("off")
                    plt.tight_layout()
                    plt.savefig(image_path)
        
                # Save the EMA model to HuggingFace Hub
                if config.push_to_hub and epoch == config.num_epochs - 1:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message="EMA model",
                    )
                    self.controlnet.push_to_hub(config.hub_model_id, variant="fp16")
                    torch.save(ema.ema_model.module.state_dict(), 'ema_model.pth')
"""
Various utilities for neural networks.

Reference:
Nichols, J., & Dhariwal, P. (2021). Improved Denoising Diffusion
Probabilistic Models. Retrieved from https://arxiv.org/abs/2102.09672

The code is adapted from the official implementation at:
https://github.com/openai/improved-diffusion/tree/main/improved_diffusion
"""

import math

import torch as th
import torch.nn as nn


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)

    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
"""
This file contains an implementation of the ADM U-Net from the paper 
"Improved Denoising Diffusion Probabilistic Models" by Nichols and
Dhariwal.

Reference:
Nichols, J., & Dhariwal, P. (2021). Improved Denoising Diffusion
Probabilistic Models. Retrieved from https://arxiv.org/abs/2102.09672

The code is adapted from the official implementation at:
https://github.com/openai/improved-diffusion/tree/main/improved_diffusion

Other UNet implementations are also included for comparison as well.
"""

from abc import abstractmethod

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DModel, UNet2DConditionModel, ControlNetModel

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1).to(self.norm.weight.dtype)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        img_resolution,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        dropout_from=16,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, (mult, num_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            if ds < dropout_from:
                dropout = 0
            
            for _ in range(num_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, (mult, num_blocks) in list(enumerate(zip(channel_mult, num_res_blocks)))[::-1]:
            for i in range(num_blocks + 1):
                if ds < dropout_from:
                    dropout = 0
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
    
#----------------------------------------------------------------------------
# End-to-end compression model as described in the paper 
# "Simple Diffusion: End-to-End Diffusion for High Resolution Images"

class simpleUNet(nn.Module):
    def __init__(self,
        img_resolution,                         # Image resolution at input/output.
        in_channels,                            # Number of color channels at input.
        out_channels,                           # Number of color channels at output.

        model_channels          = 192,          # Base multiplier for the number of channels.
        channel_mult            = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        num_res_blocks          = [2,2,2,2],    # Number of residual blocks per resolution.
        attention_resolutions   = [32,16,8],    # List of resolutions with self-attention.
        dropout                 = 0.10,         # List of resolutions with self-attention.
        dropout_from            = 16,            # Start applying dropout from this downsample onwards.

        downsample              = 4,            # Downsample factor for the initial convolution layer.

        fp16                    = False,        # Use float16 precision.
    ):
        super().__init__()

        # Initial convolution layer to downsample the input image by a factor provided
        # Assumes that the input image is square and divisible by the downsample factor
        self.conv_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=downsample,
            stride=downsample,
            bias=False,
        )

        self.conv_up = self.conv_up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=downsample,
            stride=downsample,
            bias=False,
        )

        # ADM backbone
        self.unet = UNetModel(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            dropout_from=dropout_from,
            use_fp16=fp16,
        )

    def forward(self, x, noise_labels):
        # Downsample the input image
        x = self.conv_down(x)

        # Pass through the UNet model
        x = self.unet(x, noise_labels)

        # Upsample the output image
        x = self.conv_up(x)

        return x


#----------------------------------------------------------------------------
# Adaptation of HuggingFace's UNet2DModel to use with the 
# simpleDiffusion pipeline

class UNet2D(UNet2DModel):
    def __init__(        
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
        ):

        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            time_embedding_type=time_embedding_type,
            freq_shift=freq_shift,
            flip_sin_to_cos=flip_sin_to_cos,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            mid_block_scale_factor=mid_block_scale_factor,
            downsample_padding=downsample_padding,
            downsample_type=downsample_type,
            upsample_type=upsample_type,
            dropout=dropout,
            act_fn=act_fn,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            attn_norm_num_groups=attn_norm_num_groups,
            norm_eps=norm_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=add_attention,
            class_embed_type=class_embed_type,
            num_class_embeds=num_class_embeds,
            num_train_timesteps=num_train_timesteps,
        )


    def forward(self, x, noise_labels):
        x = super().forward(x, noise_labels, return_dict=False)
        return x[0]


#----------------------------------------------------------------------------
# Adaptation of HuggingFace's UNet2DConditionModel to use with the 
# simpleDiffusion pipeline. No text conditioning is used in this case.

class UNetCondition2D(UNet2DConditionModel):
    def __init__(        
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,):

        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )


    def forward(self, x, noise_labels, downblock_additional_residuals=None, midblock_additional_residuals=None):
        encoder_hidden_states = None
        x = super().forward(
            x, 
            noise_labels, 
            encoder_hidden_states, 
            down_block_additional_residuals=downblock_additional_residuals,
            mid_block_additional_residual=midblock_additional_residuals,
            return_dict=False
            )
        return x[0]
    
#----------------------------------------------------------------------------
# Adaptation of HuggingFace's ControlNet Model to use with the
# simpleDiffusion pipeline

class ControlUNet(ControlNetModel):
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
    ):
        super().__init__(
            in_channels=in_channels,
            conditioning_channels=conditioning_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            global_pool_conditions=global_pool_conditions,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )


    def forward(self, x, noise_labels, conditioning):
        encoder_hidden_states = None
        x = super().forward(x, noise_labels, encoder_hidden_states, conditioning, return_dict=False)
        return x[0], x[1]

"""
This file contains an implementation of the pseudocode from the paper 
"Simple Diffusion: End-to-End Diffusion for High Resolution Images" 
by Emiel Hoogeboom, Tim Salimans, and Jonathan Ho.

Reference:
Hoogeboom, E., Salimans, T., & Ho, J. (2023). 
Simple Diffusion: End-to-End Diffusion for High Resolution Images. 
Retrieved from https://arxiv.org/abs/2301.11093
"""

import torch
import torch.nn as nn
from torch.special import expm1
import math
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from pathlib import Path
import os
from tqdm import tqdm
from ema_pytorch import EMA
import matplotlib.pyplot as plt

# helper
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

class simpleDiffusion(nn.Module):
    def __init__(
        self, 
        unet,
        image_size,
        noise_size=64,
        pred_param='v', 
        schedule='shifted_cosine', 
        steps=512
    ):
        super().__init__()

        # Training objective
        assert pred_param in ['v', 'eps'], "Invalid prediction parameterization. Must be 'v' or 'eps'"
        self.pred_param = pred_param

        # Sampling schedule
        assert schedule in ['cosine', 'shifted_cosine'], "Invalid schedule. Must be 'cosine' or 'shifted_cosine'"
        self.schedule = schedule
        self.noise_d = noise_size
        self.image_d = image_size

        # Model
        assert isinstance(unet, nn.Module), "Model must be an instance of torch.nn.Module."
        self.model = unet

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_params}")

        # Steps
        self.steps = steps

    def diffuse(self, x, alpha_t, sigma_t):
        """
        Function to diffuse the input tensor x to a timepoint t with the given alpha_t and sigma_t.

        Args:
        x (torch.Tensor): The input tensor to diffuse.
        alpha_t (torch.Tensor): The alpha value at timepoint t.
        sigma_t (torch.Tensor): The sigma value at timepoint t.

        Returns:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        eps_t (torch.Tensor): The noise tensor at timepoint t.
        """
        eps_t = torch.randn_like(x)

        z_t = alpha_t * x + sigma_t * eps_t

        return z_t, eps_t

    def logsnr_schedule_cosine(self, t, logsnr_min=-15, logsnr_max=15):
        """
        Function to compute the logSNR schedule at timepoint t with cosine:

        logSNR(t) = -2 * log (tan (pi * t / 2))

        Taking into account boundary effects, the logSNR value at timepoint t is computed as:

        logsnr_t = -2 * log(tan(t_min + t * (t_max - t_min)))

        Args:
        t (int): The timepoint t.
        logsnr_min (int): The minimum logSNR value.
        logsnr_max (int): The maximum logSNR value.

        Returns:
        logsnr_t (float): The logSNR value at timepoint t.
        """
        logsnr_max = logsnr_max + math.log(self.noise_d / self.image_d)
        logsnr_min = logsnr_min + math.log(self.noise_d / self.image_d)
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))

        logsnr_t = -2 * log(torch.tan(torch.tensor(t_min + t * (t_max - t_min))))

        return logsnr_t

    def logsnr_schedule_cosine_shifted(self, t):
        """
        Function to compute the logSNR schedule at timepoint t with shifted cosine:

        logSNR_shifted(t) = logSNR(t) + 2 * log(noise_d / image_d)

        Args:
        t (int): The timepoint t.
        image_d (int): The image dimension.
        noise_d (int): The noise dimension.

        Returns:
        logsnr_t_shifted (float): The logSNR value at timepoint t.
        """
        logsnr_t = self.logsnr_schedule_cosine(t)
        logsnr_t_shifted = logsnr_t + 2 * math.log(self.noise_d / self.image_d)

        return logsnr_t_shifted
        
    def clip(self, x):
        """
        Function to clip the input tensor x to the range [-1, 1].

        Args:
        x (torch.Tensor): The input tensor to clip.

        Returns:
        x (torch.Tensor): The clipped tensor.
        """
        return torch.clamp(x, -1, 1)

    @torch.no_grad()
    def ddpm_sampler_step(self, z_t, pred, logsnr_t, logsnr_s):
        """
        Function to perform a single step of the DDPM sampler.

        Args:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        pred (torch.Tensor): The predicted value from the model (v or eps).
        logsnr_t (float): The logSNR value at timepoint t.
        logsnr_s (float): The logSNR value at the sampling timepoint s.

        Returns:
        z_s (torch.Tensor): The diffused tensor at sampling timepoint s.
        """
        c = -expm1(logsnr_t - logsnr_s)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        if self.pred_param == 'v':
            x_pred = alpha_t * z_t - sigma_t * pred
        elif self.pred_param == 'eps':
            x_pred = (z_t - sigma_t * pred) / alpha_t

        x_pred = self.clip(x_pred)

        mu = alpha_s * (z_t * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s ** 2) * c

        return mu, variance

    @torch.no_grad()
    def sample(self, x):
        """
        Standard DDPM sampling procedure. Begun by sampling z_T ~ N(0, 1)
        and then repeatedly sampling z_s ~ p(z_s | z_t)

        Args:
        x_shape (tuple): The shape of the input tensor.

        Returns:
        x_pred (torch.Tensor): The predicted tensor.
        """
        z_t = torch.randn(x.shape).to(x.device)

        # Steps T -> 1
        for t in reversed(range(1, self.steps+1)):
            u_t = t / self.steps
            u_s = (t - 1) / self.steps

            if self.schedule == 'cosine':
                logsnr_t = self.logsnr_schedule_cosine(u_t)
                logsnr_s = self.logsnr_schedule_cosine(u_s)
            elif self.schedule == 'shifted_cosine':
                logsnr_t = self.logsnr_schedule_cosine_shifted(u_t)
                logsnr_s = self.logsnr_schedule_cosine_shifted(u_s)

            logsnr_t = logsnr_t.to(x.device)
            logsnr_s = logsnr_s.to(x.device)

            pred = self.model(z_t, logsnr_t)
            mu, variance = self.ddpm_sampler_step(z_t, pred, torch.tensor(logsnr_t), torch.tensor(logsnr_s))
            z_t = mu + torch.randn_like(mu) * torch.sqrt(variance)

        # Final step
        if self.schedule == 'cosine':
            logsnr_1 = self.logsnr_schedule_cosine(1/self.steps)
            logsnr_0 = self.logsnr_schedule_cosine(0)
        elif self.schedule == 'shifted_cosine':
            logsnr_1 = self.logsnr_schedule_cosine_shifted(1/self.steps)
            logsnr_0 = self.logsnr_schedule_cosine_shifted(0)

        logsnr_1 = logsnr_1.to(x.device)
        logsnr_0 = logsnr_0.to(x.device)

        pred = self.model(z_t, logsnr_1)
        x_pred, _ = self.ddpm_sampler_step(z_t, pred, torch.tensor(logsnr_1), torch.tensor(logsnr_0))
        
        x_pred = self.clip(x_pred)

        # Convert x_pred to the range [0, 1]
        x_pred = (x_pred + 1) / 2

        return x_pred
    
    def loss(self, x):
        """
        A function to compute the loss of the model. The loss is computed as the mean squared error
        between the predicted noise tensor and the true noise tensor. Various prediction parameterizations
        imply various weighting schemes as outlined in Kingma et al. (2023)

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        loss (torch.Tensor): The loss value.
        """
        t = torch.rand(x.shape[0])

        if self.schedule == 'cosine':
            logsnr_t = self.logsnr_schedule_cosine(t)
        elif self.schedule == 'shifted_cosine':
            logsnr_t = self.logsnr_schedule_cosine_shifted(t)

        logsnr_t = logsnr_t.to(x.device)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        z_t, eps_t = self.diffuse(x, alpha_t, sigma_t)
        pred = self.model(z_t, logsnr_t)

        if self.pred_param == 'v':
            eps_pred = sigma_t * z_t + alpha_t * pred
        else: 
            eps_pred = pred

        # Apply min-SNR weighting (https://arxiv.org/pdf/2303.09556)
        snr = torch.exp(logsnr_t).clamp_(max = 5)
        if self.pred_param == 'v':
            weight = 1 / (1 + snr)
        else:
            weight = 1 / snr

        weight = weight.view(-1, 1, 1, 1)

        loss = torch.mean(weight * (eps_pred - eps_t) ** 2)

        return loss

    def train_loop(self, config, optimizer, train_dataloader, lr_scheduler):
        """
        A function to train the model.

        Args:
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        """
        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_dir=os.path.join(config.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            if config.push_to_hub:
                repo_id = create_repo(
                    repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
                ).repo_id
                print(repo_id)

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare( 
            self.model, optimizer, train_dataloader, lr_scheduler
        )

        # Create an EMA model
        ema = EMA(
            model,
            beta=0.9999,
            update_after_step=100,
            update_every=10
        )

        global_step = 0

        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                x = batch["images"]

                with accelerator.accumulate(model):
                    loss = self.loss(x)
                    loss = loss.to(next(model.parameters()).dtype)
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Update EMA model parameters
                ema.update()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images
            if accelerator.is_main_process:
                self.model = accelerator.unwrap_model(model)
                self.model.eval()

            # Make directory for saving images
                os.makedirs(os.path.join(config.output_dir, "images"), exist_ok=True)

                if epoch % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    sample = self.sample(x[0].unsqueeze(0))
                    sample = sample.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    image_path = os.path.join(config.output_dir, "images", f"sample_{epoch}.png")
                    plt.imsave(image_path, sample[0])
        
                # Save the EMA model to HuggingFace Hub
                if config.push_to_hub and epoch == config.num_epochs - 1:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message="EMA model",
                    )
                    self.model.push_to_hub(config.hub_model_id, variant="fp16")
                    torch.save(ema.ema_model.module.state_dict(), 'ema_model.pth')
"""
Helpers to train with 16-bit precision.

Reference:
Nichols, J., & Dhariwal, P. (2021). Improved Denoising Diffusion
Probabilistic Models. Retrieved from https://arxiv.org/abs/2102.09672

The code is adapted from the official implementation at:
https://github.com/openai/improved-diffusion/tree/main/improved_diffusion
"""

import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        l.bias.data = l.bias.data.float()


def make_master_params(model_params):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    model_params = list(model_params)

    for param, master_param in zip(
        model_params, unflatten_master_params(model_params, master_params)
    ):
        param.detach().copy_(master_param)


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), tuple(tensor for tensor in model_params))


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
"""
A script for training a ControlNet on the Smithsonian Butterflies dataset.

An adaptation of the HuggingFace ControlNet training script.
"""

from diffusion.unet import UNetCondition2D, ControlUNet
from diffusion.control_net import ControlNet
from utils.canny import AddCannyImage

from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np


class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 4
    num_epochs = 1000
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    lr_warmup_steps = 10000
    save_image_epochs = 50
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_model_id = "faverogian/Smithsonian128ControlNet"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


def main():
    
    config = TrainingConfig

    dataset_name = "huggan/smithsonian_butterflies_subset"

    dataset = load_dataset(dataset_name, split="train")

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        conditions = [AddCannyImage()(image) for image in images]
        return {"images":images, "conditions":conditions}

    dataset.set_transform(transform)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    unet = UNetCondition2D.from_pretrained("faverogian/Smithsonian128UNet", variant="fp16")
    controlnet = ControlUNet.from_unet(unet, conditioning_channels=1)

    optimizer = torch.optim.Adam(controlnet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
    )

    controlnet_model = ControlNet(
        unet=unet,
        controlnet=controlnet,
        image_size=config.image_size
    )

    controlnet_model.train_loop(
        config=config,
        optimizer=optimizer,
        train_dataloader=train_loader,
        lr_scheduler=lr_scheduler
    )


if __name__ == '__main__':
    main()
