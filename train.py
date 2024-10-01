from torch.utils.data import DataLoader
from torchvision import transforms
from HumanDataset import SyntheticHumanDataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils import save_image_grid,save_image_grid_4
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Utility functions
def generate_camera_rays(height, width, focal_length):
    x, y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
    x = (x - width * 0.5) / focal_length
    y = -(y - height * 0.5) / focal_length
    z = -torch.ones_like(x)
    directions = torch.stack([x, y, z], dim=-1)
    return F.normalize(directions, dim=-1)

def render_volume(rgb, density, t_vals):
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    delta = torch.cat([delta, torch.tensor([1e10]).expand(delta[..., :1].shape)], dim=-1)
    alpha = 1 - torch.exp(-density * delta)
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * t_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)
    return rgb_map, depth_map, acc_map

def compute_normal_loss(density, points):
    print(f"density shape: {density.shape}, requires_grad: {density.requires_grad}")
    print(f"points shape: {points.shape}, requires_grad: {points.requires_grad}")
    
    try:
        grads = torch.autograd.grad(density.sum(), points, create_graph=True, allow_unused=True, retain_graph=True)
        if grads[0] is None:
            print("👹 Warning: grad_density is None. Checking if density is connected to points...")
            test_loss = (density * points.sum()).sum()
            test_grads = torch.autograd.grad(test_loss, points, retain_graph=True)
            if test_grads[0] is None:
                print("Error: density is not connected to points in the computational graph.")
            else:
                print("density is connected to points, but gradients are not flowing as expected.")
            return torch.tensor(0.0, device=density.device, requires_grad=True)
        
        grad_density = grads[0]
        normal = F.normalize(grad_density, dim=-1)
        return (normal[:, 1:] - normal[:, :-1]).square().mean()
    except Exception as e:
        print(f"Error in compute_normal_loss: {e}")
        return torch.tensor(0.0, device=density.device, requires_grad=True)


def train_en3d(generator, discriminator, dataloader, num_epochs, device, render_function):
    # Ensure the generator's output size matches the rendered images
    generator.output_size = 256  # Set this to match the size of rendered_images
    generator.num_points = 64 * 64  # 4096 points, which is a perfect square

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    global_step = 0
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            vertices = batch['vertices'].to(device)
            faces = batch['faces'].to(device)
            textures = batch['texture'].to(device)
            sample_ids = batch['id']

            batch_size = vertices.size(0)

            # Generate random latent vectors
            z = torch.randn(batch_size, generator.latent_dim).to(device)
            
            # Generate random camera positions
            camera_positions = generate_random_cameras(batch_size).to(device)
            
            # Render synthetic views using the provided 3D models and textures
            rendered_images = render_function(vertices, faces, textures, camera_positions)
            
            # Generate 3D-aware images using the generator
            generated_images, generated_densities, points = generator(z, camera_positions)
            
            # Train discriminator
            optimizer_D.zero_grad()
            d_real = discriminator(rendered_images)
            d_fake = discriminator(generated_images.detach())
            d_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real)) + \
                     F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
            d_loss.backward()
            optimizer_D.step()
            
            # Train generator
            optimizer_G.zero_grad()
            d_fake = discriminator(generated_images)
            g_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
            
            # Add reconstruction loss
            recon_loss = F.mse_loss(generated_images, rendered_images)
            
            # Add normal consistency loss (if implemented)
            normal_loss = compute_normal_loss(generated_densities, points)  # Use points from the generator
            
            # Total generator loss
            total_g_loss = g_loss + 0.1 * recon_loss + 0.01 * normal_loss
            
            total_g_loss.backward()
            optimizer_G.step()
        
            pbar.set_postfix({
                'D Loss': f"{d_loss.item():.4f}",
                'G Loss': f"{g_loss.item():.4f}",
                'Recon Loss': f"{recon_loss.item():.4f}",
                'Normal Loss': f"{normal_loss.item():.4f}"
            })
            global_step += 1
           
         # Save samples periodically
            if global_step % 10 == 0:
                with torch.no_grad():
                    fixed_z = torch.randn(16, generator.latent_dim).to(device)
                    fixed_cameras = generate_random_cameras(16).to(device)
                    
                    # Generate images
                    generated_images, _, _ = generator(fixed_z, fixed_cameras)
                    
                    # Render images from the batch
                    rendered_images = render_function(
                        batch['vertices'][:16].to(device),
                        batch['faces'][:16].to(device),
                        batch['texture'][:16].to(device),
                        fixed_cameras
                    )
                    
                    # Get sample IDs
                    # sample_ids = batch['id'][:16] if 'id' in batch else None
                    save_image_grid_4(generated_images,'test.png')  
                     
                    # # Save the image grid
                    # save_image_grid(
                    #     generated_images,
                    #     rendered_images,
                    #     f"samples_epoch_{epoch}_batch_{global_step}.png",
                    #     sample_ids=sample_ids
                    # )
        # Optional: Save generated samples and model checkpoints
        if (epoch + 1) % 10 == 0:
            # save_samples(generator, sample_ids, epoch, device)
            save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch)

    

def generate_random_cameras(batch_size):
    # Generate random camera positions
    # This is a simplified version; you might want to use more sophisticated camera generation
    theta = np.random.uniform(0, 2 * np.pi, batch_size)
    phi = np.random.uniform(0, np.pi, batch_size)
    r = np.random.uniform(1.5, 2.5, batch_size)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    return torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)



def save_samples(generator, sample_ids, epoch, device):
    # Save some generated samples for visualization
    with torch.no_grad():
        z = torch.randn(16, generator.latent_dim).to(device)
        cameras = generate_random_cameras(16).to(device)
        generated_images, _, _ = generator(z, cameras)

        
    # Save the generated images (implement this function based on your needs)
    save_image_grid(generated_images, f"samples_epoch_{epoch}.png", sample_ids[:16])

def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch):
    # Save model checkpoints
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'epoch': epoch
    }, f"checkpoint_epoch_{epoch}.pth")



def move_to_device(module, device):
    for param in module.parameters():
        param.data = param.data.to(device)
    for buffer in module.buffers():
        buffer.data = buffer.data.to(device)
    for child in module.children():
        move_to_device(child, device)

from torch.utils.data import DataLoader
from torchvision import transforms
from Model import Discriminator,En3DGenerator,RenderFunction
# import your_renderer_module  # Import your chosen renderer (e.g., pytorch3d or nvdiffrast)



# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    latent_dim = 512
    w_dim = 512
    feature_dim = 32
    num_epochs = 100
    batch_size = 32

    # Define transforms for the texture images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    
    # Initialize models
    generator = En3DGenerator(latent_dim, w_dim, feature_dim,device).to(device)
    discriminator = Discriminator().to(device)
    
    # Load data
    # Create dataset and dataloader
    dataset = SyntheticHumanDataset('./synthetic_data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)    

    # Initialize your models
    discriminator = Discriminator().to(device)
    render_function = RenderFunction(image_size=256).to(device)

    # Train the model
    train_en3d(generator, discriminator, dataloader, num_epochs=100, device=device, render_function=render_function)

