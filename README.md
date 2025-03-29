# Introduction
This repository is primarily for fine-tuning and training the SAM (Segment Anything Model) to segment specific objects. Test images are sourced from ImageNet.

# Dependency Installation and Virtual Environment Setup
```bash
# create enviroment
python -m venv sam
source sam/bin/activate
# dependency installation
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python matplotlib
```
# Downloading the SAM Model
There are three available model checkpoints. You can download them from the official repository:
[🔗 SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)

# Quickly Obtain Key Points from an Image
To extract image coordinates easily, you can use the following online tool:
[🔗 Image Coordinate Extractor](https://uutool.cn/img-coord/)

