import torch
from segment_anything import sam_model_registry, SamPredictor
# set model type and checkpoint
sam_checkpoint = "sam_vit_h.pth"  # replace with your checkpoint path
model_type = "vit_h"  # option with  "vit_b"、"vit_l"、"vit_h"

# load sam model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

# create Predictor
predictor = SamPredictor(sam)