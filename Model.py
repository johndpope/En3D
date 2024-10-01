import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import nvdiffrast.torch as dr


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, w_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, w_dim)
        )

    def forward(self, z):
        return self.net(z)

class TriplaneGenerator(nn.Module):
    def __init__(self, w_dim, feature_dim):
        super().__init__()
        self.w_dim = w_dim
        self.feature_dim = feature_dim

        self.plane_generators = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(w_dim, 512, 4, 1, 0),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, feature_dim, 4, 2, 1),
            ) for _ in range(3)
        ])

    def forward(self, w):
        planes = []
        for gen in self.plane_generators:
            with torch.no_grad():
                plane = gen(w.view(-1, self.w_dim, 1, 1))
            planes.append(plane.requires_grad_())
        return planes

class PatchComposedRenderer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.to_rgb = nn.Conv2d(feature_dim, 3, 1)
        self.to_density = nn.Conv2d(feature_dim, 1, 1)

    def forward(self, planes, points):
        xy, xz, yz = planes
        batch_size = xy.size(0)
        N_points = points.shape[1]
        H_out = W_out = int(np.sqrt(N_points))
        if H_out * W_out != N_points:
            raise ValueError("Number of points must be a perfect square")

   
        # Use grid_sample correctly

        feat_xy = F.interpolate(xy, size=(H_out, W_out), mode='bilinear', align_corners=True)
        feat_xz = F.interpolate(xz, size=(H_out, W_out), mode='bilinear', align_corners=True)
        feat_yz = F.interpolate(yz, size=(H_out, W_out), mode='bilinear', align_corners=True)



        features = feat_xy + feat_xz + feat_yz
        rgb = self.to_rgb(features)
        density = self.to_density(features).sigmoid()

        return rgb, density

class GeometricSculpting(nn.Module):
    def __init__(self, resolution=256):
        super().__init__()
        self.resolution = resolution
        self.voxel_grid = nn.Parameter(torch.randn(1, 1, resolution, resolution, resolution))
        self.refine_net = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 1, 3, padding=1)
        )

    def forward(self, points):
        batch_size = points.shape[0]
        # Ensure points are in the range [-1, 1]
        points = points.clamp(-1, 1)
        
        # Reshape points to [batch_size, num_points, 1, 1, 3]
        points = points.unsqueeze(2).unsqueeze(2)
        
        # Expand voxel grid to match batch size
        voxel_grid_expanded = self.voxel_grid.expand(batch_size, -1, -1, -1, -1)
        
        # Sample from the voxel grid
        with torch.no_grad():
            samples = F.grid_sample(voxel_grid_expanded, points, align_corners=True)
        
        # Refine the sampled values
        refined = self.refine_net(samples)
        
        # Reshape to [batch_size, num_points]
        return refined.squeeze(1).squeeze(-1).squeeze(-1)

    def __repr__(self):
        return f"GeometricSculpting(resolution={self.resolution})"

class ExplicitTexturing(nn.Module):
    def __init__(self, texture_size=1024, num_parts=5):
        super().__init__()
        self.texture_size = texture_size
        self.num_parts = num_parts
        self.textures = nn.ParameterList([
            nn.Parameter(torch.rand(1, 3, texture_size, texture_size))
            for _ in range(num_parts)
        ])
        self.uv_partitioning = nn.Conv2d(2, num_parts, 1)

    def forward(self, uv_coords):
        partitions = self.uv_partitioning(uv_coords.permute(0, 3, 1, 2)).softmax(dim=1)
        textures = torch.stack([
            F.grid_sample(texture, uv_coords, align_corners=True).squeeze(0)
            for texture in self.textures
        ])
        return (textures * partitions.unsqueeze(2)).sum(dim=0)
    


class En3DGenerator(nn.Module):
    def __init__(self, latent_dim=512, w_dim=512, feature_dim=32, num_points=4096, output_size=256, device='cuda'):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.output_size = output_size
        self.mapping_network = MappingNetwork(latent_dim, w_dim)
        self.triplane_generator = TriplaneGenerator(w_dim, feature_dim)
        self.renderer = PatchComposedRenderer(feature_dim)
        self.geometric_sculpting = GeometricSculpting()
        self.explicit_texturing = ExplicitTexturing()

    def forward(self, z, camera_positions):
        # Debug prints
        print(f"Debug - Input shapes: z {z.shape}, camera_positions {camera_positions.shape}")
        
        # Step 1: Pass through mapping network
        batch_size = z.size(0)
        w = self.mapping_network(z)
        print(f"Debug - Mapping network output shape: {w.shape}")
        
        # Step 2: Generate triplanes
        planes = self.triplane_generator(w)
        print(f"Debug - Triplane generator output shapes: xy {planes[0].shape}, xz {planes[1].shape}, yz {planes[2].shape}")
        
        # Step 3: Generate points
        points = self.generate_points(batch_size)
        points = points.requires_grad_()  # Ensure points require gradients for backprop
        print(f"Debug - Generated points shape: {points.shape}")
        
        # Step 4: Render rgb and density
        rgb, density = self.renderer(planes, points)
        print(f"Debug - Renderer output shapes: rgb {rgb.shape}, density {density.shape}")

        # Step 5: Geometric sculpting to refine density
        sculpted_density = self.geometric_sculpting(points)
        print(f"Debug - Sculpted density shape: {sculpted_density.shape}")
        
        # Step 6: Reshape sculpted_density to match the shape of density
        # Reshape sculpted_density to match the spatial dimensions of density
        sculpted_density_reshaped = sculpted_density.view(batch_size, 1, density.shape[2], density.shape[3])
        print(f"Debug - Reshaped sculpted density shape: {sculpted_density_reshaped.shape}")

        # Step 7: Final density calculation
        final_density = density * sculpted_density_reshaped
        print(f"Debug - Final density shape: {final_density.shape}")
        
        # Step 8: Interpolate rgb output to the desired size
        rgb_reshaped = F.interpolate(rgb, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        print(f"Debug - Reshaped rgb shape: {rgb_reshaped.shape}")

        return rgb_reshaped, final_density, points

    def generate_points(self, batch_size):
        # Generate a grid of points in the unit cube [-1, 1] x [-1, 1] x [-1, 1]
        side_length = int(np.sqrt(self.num_points))
        x = torch.linspace(-1, 1, side_length, device=self.device)
        y = torch.linspace(-1, 1, side_length, device=self.device)
        z = torch.linspace(-1, 1, side_length, device=self.device)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([xx, yy, zz], dim=-1).view(1, -1, 3)
        return points.expand(batch_size, -1, -1)



    def __repr__(self):
        return f"En3DGenerator(latent_dim={self.latent_dim}, num_points={self.num_points}, output_size={self.output_size})"

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)




class RenderFunction(nn.Module):
    def __init__(self, image_size=256):
        super(RenderFunction, self).__init__()
        self.image_size = image_size
        self.ctx = dr.RasterizeGLContext()

    def forward(self, vertices, faces, textures, camera_positions):
        batch_size = vertices.shape[0]
        device = vertices.device

        print(f"Input shapes: vertices {vertices.shape}, faces {faces.shape}, textures {textures.shape}, camera_positions {camera_positions.shape}")

        # Ensure faces have the correct shape
        if faces.dim() == 3:
            faces = faces.reshape(-1, 3)
        elif faces.dim() != 2 or faces.shape[1] != 3:
            raise ValueError(f"Faces tensor has incorrect shape: {faces.shape}. Expected shape: [num_faces, 3]")

        # Ensure textures have the correct shape for interpolation
        if textures.dim() == 4:  # [batch_size, channels, height, width]
            textures = textures.permute(0, 2, 3, 1)  # Change to [batch_size, height, width, channels]
        elif textures.dim() == 3:  # [height, width, channels]
            textures = textures.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError(f"Textures tensor has incorrect shape: {textures.shape}. Expected shape: [batch_size, height, width, channels] or [height, width, channels]")

        # Reshape textures to [num_vertices, channels]
        num_vertices = vertices.shape[1]
        textures = textures.view(-1, textures.shape[-1])[:num_vertices]

        print(f"Shapes after reshaping: vertices {vertices.shape}, faces {faces.shape}, textures {textures.shape}")

        # Ensure all tensors are contiguous
        vertices = vertices.contiguous()
        faces = faces.contiguous()
        textures = textures.contiguous()

        print(f"Faces shape after reshaping: {faces.shape}")

        # Create perspective projection matrix
        proj_mtx = self.perspective_projection(60, 1.0, 0.1, 100).to(device)
        print(f"Projection matrix shape: {proj_mtx.shape}")

        # Apply perspective projection
        vertices_proj = self.apply_perspective(vertices, proj_mtx, camera_positions)
        print(f"Projected vertices shape: {vertices_proj.shape}")

        # Prepare vertices for nvdiffrast (clip space)
        vertices_clip = torch.cat([vertices_proj, torch.ones_like(vertices_proj[..., :1])], dim=-1)
        vertices_clip[..., :2] = -vertices_clip[..., :2]
        vertices_clip[..., 2] = vertices_clip[..., 2] * 2 - 1  # Map z from [0, 1] to [-1, 1]
        vertices_clip = vertices_clip.contiguous()
        print(f"Clip space vertices shape: {vertices_clip.shape}")

        # Ensure faces is of int32 type and contiguous
        faces = faces.to(torch.int32).contiguous()

        # Rasterize
        rast, _ = dr.rasterize(self.ctx, vertices_clip, faces, resolution=[self.image_size, self.image_size])
        rast = rast.contiguous()
        print(f"Rasterization output shape: {rast.shape}")

        # Interpolate features (textures)
        feature_maps, _ = dr.interpolate(textures, rast, faces)
        print(f"Interpolated feature maps shape: {feature_maps.shape}")

        # Compute and interpolate normals
        normals = self.compute_normals(vertices, faces)
        normals = normals.contiguous()
        print(f"Computed normals shape: {normals.shape}")
        interpolated_normals, _ = dr.interpolate(normals, rast, faces)
        print(f"Interpolated normals shape: {interpolated_normals.shape}")

        # Compute lighting
        light_dir = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 1, 3).expand_as(interpolated_normals)
        diffuse = torch.sum(interpolated_normals * light_dir, dim=-1, keepdim=True).clamp(min=0)
        print(f"Diffuse lighting shape: {diffuse.shape}")

        # Apply diffuse shading to feature maps
        shaded_features = feature_maps * diffuse
        print(f"Shaded features shape: {shaded_features.shape}")

        # Rearrange to [batch_size, C, H, W]
        shaded_features = shaded_features.permute(0, 3, 1, 2).contiguous()
        print(f"Final output shape: {shaded_features.shape}")

        return shaded_features

    def perspective_projection(self, fov, aspect_ratio, near, far):
        fov_rad = np.radians(fov)
        f = 1 / np.tan(fov_rad / 2)
        proj_matrix = torch.tensor([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=torch.float32)
        return proj_matrix

    def apply_perspective(self, vertices, proj_mtx, camera_positions):
        # Apply camera translation
        vertices = vertices - camera_positions.unsqueeze(1)

        # Convert to homogeneous coordinates
        vertices_hom = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)

        # Apply projection matrix
        vertices_proj = torch.matmul(vertices_hom, proj_mtx.T)

        # Perspective division
        vertices_proj = vertices_proj[..., :3] / vertices_proj[..., 3:]

        return vertices_proj

    def compute_normals(self, vertices, faces):
        batch_size, num_vertices, _ = vertices.shape
        
        # Expand faces to match batch size and convert to int64
        faces_expanded = faces.unsqueeze(0).expand(batch_size, -1, -1).to(torch.int64)
        
        # Extract face indices
        f0, f1, f2 = faces_expanded[:, :, 0], faces_expanded[:, :, 1], faces_expanded[:, :, 2]
        
        # Gather vertex positions for each face
        v0 = vertices.gather(1, f0.unsqueeze(-1).expand(-1, -1, 3))
        v1 = vertices.gather(1, f1.unsqueeze(-1).expand(-1, -1, 3))
        v2 = vertices.gather(1, f2.unsqueeze(-1).expand(-1, -1, 3))
        
        # Compute normals
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        normals = F.normalize(normals, dim=-1)
        
        return normals