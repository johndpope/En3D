import os
import torch
import trimesh
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SyntheticHumanDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for subdir in os.listdir(self.data_dir):
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.isdir(subdir_path):
                obj_file = None
                glb_file = None
                png_file = None
                
                for file in os.listdir(subdir_path):
                    if file.endswith('.obj'):
                        obj_file = os.path.join(subdir_path, file)
                    elif file.endswith('.glb'):
                        glb_file = os.path.join(subdir_path, file)
                    elif file == 'body.png':
                        png_file = os.path.join(subdir_path, file)
                
                if (obj_file or glb_file) and png_file:
                    samples.append({
                        'model': obj_file if obj_file else glb_file,
                        'texture': png_file,
                        'id': subdir
                    })
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        model_path = sample['model']
        texture_path = sample['texture']
        sample_id = sample['id']

        # Load 3D model
        mesh = trimesh.load(model_path)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        
        print(f"Loaded faces shape: {faces.shape}")  # Debug print
        assert faces.dim() == 2 and faces.shape[1] == 3, f"Unexpected faces shape: {faces.shape}"
       

        print("texture_path:",texture_path)
        # Load texture
        # Load texture
        texture = Image.open(texture_path).convert('RGB')
        if self.transform:
            texture = self.transform(texture)
        else:
            texture = transforms.ToTensor()(texture)
        
        # Ensure texture is in the correct format [channels, height, width]
        if texture.dim() == 2:
            texture = texture.unsqueeze(0)
        elif texture.dim() == 3 and texture.shape[0] != 3:
            texture = texture.permute(2, 0, 1)
        

      
        
        return {
            'vertices': vertices.contiguous(),
            'faces': faces.contiguous(),
            'texture': texture.contiguous(),
            'id': sample_id
        }

    def get_sample_info(self, idx):
        return self.samples[idx]