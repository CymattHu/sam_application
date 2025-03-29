# Introduction
this repo is mainly to fine tune and train sam model to segment specific object. test images come from ImageNet.

# Dependency installation and virtual enviroment creation
```bash
# create enviroment
python -m venv sam
source sam/bin/activate
# dependency installation
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python matplotlib
```
# SAM model download
there are three models you can choose , download link is here https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

# Key point fast acquire method from image
you can get image coodinate by use this website https://uutool.cn/img-coord/

