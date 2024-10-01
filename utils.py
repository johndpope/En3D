import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


import torchvision.utils as vutils

def save_image_grid_4(images, filename, sample_ids=None):
    # Unnormalize images if necessary
    # images = images * 0.5 + 0.5  # If images were normalized to [-1, 1]
    
    grid = vutils.make_grid(images, nrow=4, normalize=True, scale_each=True)
    vutils.save_image(grid, filename)


def save_image_grid(generated_images, rendered_images, filename, sample_ids=None, nrow=4):
    """
    Save a grid of images, including both generated and rendered images.
    
    Args:
    - generated_images (torch.Tensor): Tensor of generated images (B, C, H, W)
    - rendered_images (torch.Tensor): Tensor of rendered images from 3D models (B, C, H, W)
    - filename (str): Output filename
    - sample_ids (list): List of sample IDs corresponding to the images
    - nrow (int): Number of images per row in the grid
    """
    # Ensure the tensors are on CPU and in the correct format
    generated_images = generated_images.cpu()
    rendered_images = rendered_images.cpu()
    
    # Denormalize if necessary (assuming images are in range [-1, 1])
    generated_images = (generated_images + 1) / 2
    rendered_images = (rendered_images + 1) / 2
    
    # Clamp values to [0, 1] range
    generated_images = torch.clamp(generated_images, 0, 1)
    rendered_images = torch.clamp(rendered_images, 0, 1)
    
    # Create a grid of images
    generated_grid = torchvision.utils.make_grid(generated_images, nrow=nrow, padding=2, normalize=False)
    rendered_grid = torchvision.utils.make_grid(rendered_images, nrow=nrow, padding=2, normalize=False)
    
    # Combine generated and rendered grids vertically
    combined_grid = torch.cat([generated_grid, rendered_grid], dim=1)
    
    # Convert to PIL Image
    img = torchvision.transforms.ToPILImage()(combined_grid)
    
    # Add labels if sample_ids are provided
    if sample_ids is not None:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        for i, sample_id in enumerate(sample_ids):
            x = (i % nrow) * (generated_images.shape[3] + 2)
            y_generated = (i // nrow) * (generated_images.shape[2] + 2)
            y_rendered = y_generated + generated_grid.shape[1] + 2
            
            draw.text((x+5, y_generated+5), f"Gen {sample_id}", (255, 255, 255), font=font)
            draw.text((x+5, y_rendered+5), f"Ren {sample_id}", (255, 255, 255), font=font)
    
    # Save the image
    img.save(filename)
    print(f"Saved image grid to {filename}")

    # Optionally, you can also log some statistics about the images
    print(f"Generated images - Min: {generated_images.min():.4f}, Max: {generated_images.max():.4f}, Mean: {generated_images.mean():.4f}")
    print(f"Rendered images - Min: {rendered_images.min():.4f}, Max: {rendered_images.max():.4f}, Mean: {rendered_images.mean():.4f}")
