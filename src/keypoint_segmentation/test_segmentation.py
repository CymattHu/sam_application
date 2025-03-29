import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import os
import torch.optim as optim
import matplotlib.pyplot as plt
# set model type and checkpoint
sam_checkpoint = "sam_vit_h_4b8939.pth"  # replace with your checkpoint path
model_type = "vit_h"  # option with  "vit_b"、"vit_l"、"vit_h"

# load sam model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

# create Predictor
predictor = SamPredictor(sam)

# read image

img_path = os.path.dirname(os.path.abspath(__file__))+"/../../test_image/um_port_234.jpg"  # replace with your image folder path

print("img_path:", img_path)

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Image Embeding
predictor.set_image(image)


input_point = np.array([[448, 324]])  # set the coordinates of the point (x, y)
input_label = np.array([1])  # 1 for foreground, 0 for background

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  # generate multiple masks
)

# select the best mask based on the score
best_mask = masks[np.argmax(scores)]



plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.imshow(best_mask, alpha=0.5, cmap='jet')
plt.axis("off")
plt.show()