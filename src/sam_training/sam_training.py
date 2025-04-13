from segment_anything import sam_model_registry
import torch.optim as optim
import torch
from data_loader import SegmentationDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import os

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
dataset = SegmentationDataset(
    image_dir=os.path.dirname(os.path.abspath(__file__))+"/../../segmentation_dataset/image",
    mask_dir=os.path.dirname(os.path.abspath(__file__))+"/../../segmentation_dataset/mask",
    annotation_file=os.path.dirname(os.path.abspath(__file__))+"/../../segmentation_dataset/annotations.json"
)

dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

for param in sam.parameters():
    param.requires_grad = False
for param in sam.mask_decoder.parameters():
    param.requires_grad = True

# create new mask decoder
optimizer = optim.AdamW(sam.mask_decoder.parameters(), lr=1e-4)

transform = transforms.Compose([
    transforms.ToTensor(),  # conver PIL image to tensor
])

# loss function
criterion = torch.nn.CrossEntropyLoss()
device = "cuda" if torch.cuda.is_available() else "cpu"
for epoch in range(4):
    for images, masks in dataloader:
        # images = images.to(device)
        # masks = masks.to(device)
        # # batched_input = {"image": images,"original_size": (images.shape[1], images.shape[2])}
        # # forward propagation
        # output = sam(images,multimask_output=False)  # mask prediction
        # output_masks = output[0]["masks"].float()  # get mask of the predicrtion
        # # calculate loss
        # loss = criterion(output_masks, masks.long())
        
        # # backward propagation
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        images, masks = images.to(device), masks.to(device)
        batched_input = [{"image": images, "original_size": (images.shape[-2], images.shape[-1])}]
        
        # foward propagation
        output_masks = sam(batched_input,multimask_output=False)  # predict masks
        
        # calculate loss
        loss = criterion(output_masks, masks)
        
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
torch.save(sam.state_dict(), "finetuned_sam.pth")