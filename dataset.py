import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class PelletDataset(Dataset):
    def __init__(self, img_dir, density_dir, file_list, transform=None):
        """
        Args:
            img_dir (str): Path to images folder
            density_dir (str): Path to density maps folder
            file_list (str): Path to a text file listing image filenames
            transform: Optional image transforms (e.g. augmentations)
        """
        self.img_dir = img_dir
        self.density_dir = density_dir
        self.transform = transform

        # Load list of images to use (train or val)
        with open(file_list, "r") as f:
            self.image_names = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        den_path = os.path.join(self.density_dir, img_name.replace(".jpg", ".npy"))

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB

        # Load density map
        density = np.load(den_path)

        # Normalize image to [0,1]
        img = img.astype(np.float32) / 255.0
        density = density.astype(np.float32)

        # Convert to torch tensors
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW
        density = torch.from_numpy(density).unsqueeze(0)  # Add channel dim

        if self.transform:
            img = self.transform(img)

        return img, density
