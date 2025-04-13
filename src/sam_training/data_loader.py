import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import json

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, annotation_file):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

        # load JSON annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # image filenames and mask filenames
        self.image_filenames = [ann['image_filename'] for ann in self.annotations]
        self.mask_filenames = [ann['mask_filename'] for ann in self.annotations]
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # get image and mask file paths
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        # load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 
        
        image = self.transform(image)
        mask = np.array(mask)  
        mask = torch.tensor(mask, dtype=torch.long) 
        
        return image, mask