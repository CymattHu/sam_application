import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import os
import torch.optim as optim
# set model type and checkpoint
sam_checkpoint = "sam_vit_h_4b8939.pth"  # replace with your checkpoint path
model_type = "vit_h"  # option with  "vit_b"、"vit_l"、"vit_h"

# load sam model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

# create Predictor
predictor = SamPredictor(sam)

# read image

image_folder = os.path.dirname(os.path.abspath(__file__))+"/../test_image"  # replace with your image folder path
for file_name in os.listdir(image_folder):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):  # 过滤图片格式
        img_path = os.path.join(image_folder, file_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # # Image Embeding
        # predictor.set_image(image)