from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import torch
from PIL import Image, ImageTk,ImageDraws
import numpy as np
import cv2

class COCODataset(Dataset):
    def __init__(self, image_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.image_dir = image_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = f"{self.image_dir}/{self.coco.loadImgs(img_id)[0]['file_name']}"
        image = Image.open(img_path).convert("RGB")

        # get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for ann in anns:
            for seg in ann["segmentation"]:
                pts = np.array(seg).reshape(-1, 2)
                cv2.fillPoly(mask, [pts.astype(np.int32)], color=1)

        # convert mask to tensor
        if self.transform:
            image = self.transform(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

    def __len__(self):
        return len(self.ids)

# data convertion
transform = transforms.Compose([transforms.ToTensor()])
dataset = COCODataset(image_dir="dataset/images", ann_file="dataset/annotations.json", transform=transform)