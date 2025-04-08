import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from preprocess import load_decathlon_image, load_decathlon_mask

class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')])
        self.transform = transform

        assert len(self.image_files) == len(self.mask_files), \
            "Mismatch between number of images and masks"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = load_decathlon_image(image_path)  # shape: (C, H, W, D)
        mask = load_decathlon_mask(mask_path)     # shape: (H, W, D)

        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        return image_tensor, mask_tensor
