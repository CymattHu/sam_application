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

        # 加载标注文件（假设是 JSON 格式）
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # 图像文件名（假设 JSON 标注中包含图像的文件名）
        self.image_filenames = [ann['image_filename'] for ann in self.annotations]
        self.mask_filenames = [ann['mask_filename'] for ann in self.annotations]
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # 获取图像和掩码的文件路径
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        # 加载图像和掩码
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 处理为单通道灰度图
        
        # 如果有 transform，进行预
        image = self.transform(image)
        mask = np.array(mask)  # 转为 NumPy 数组，方便后面处理
        mask = torch.tensor(mask, dtype=torch.long)  # 转为 tensor 格式
        
        return image, mask