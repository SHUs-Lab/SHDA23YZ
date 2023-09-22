# Prerequisites
PyTorch compatible GPU with CUDA 12.1 Conda (Python 3.9)
Other packages are listed in requirements.txt.
# Data
The data is not big. It contains $1600$ images in the train folder and $400$ images in the test folder. 
Firstly, download and unzip the [data.zip](https://drive.google.com/file/d/1kjPuhyrEgeIA5uaFoMgv4qlC-9jQzK47/view?usp=sharing).

# Usage
## Create a new conda environment:
   ```
   conda create --name culvert python=3.9
   activate culvert       # Windows
   conda activate culvert # Linux
   ```
## Clone this repo:
   ```
   git clone https://github.com/SHUs-Lab/SHDA23YZ.git
   cd SHDA23YZ
   ```
## Install required packages by typing
   ```
   pip install -r requirements.txt
   ```
## Download [data.zip](https://drive.google.com/file/d/1kjPuhyrEgeIA5uaFoMgv4qlC-9jQzK47/view?usp=sharing) and unzip it.
Then, run object_detection.py to train SPPNet.
Also, you can run sppnet_search.py to search for neural architecture.

# SPPNet parameters are defined below:

## Model Summary

| Layer (type:depth-idx)                | Param #         |
|---------------------------------------|-----------------|
| SPPNet                                |                 |
| ├─Sequential: 1-1                    |                 |
| │   └─Conv2d: 2-1                    | 1,792           |
| │   └─ReLU: 2-2                      |                 |
| │   └─MaxPool2d: 2-3                 |                 |
| │   └─Conv2d: 2-4                    | 73,856          |
| │   └─ReLU: 2-5                      |                 |
| │   └─MaxPool2d: 2-6                 |                 |
| │   └─Conv2d: 2-7                    | 295,168         |
| │   └─ReLU: 2-8                      |                 |
| │   └─MaxPool2d: 2-9                 |                 |
| ├─ModuleList: 1-2                    |                 |
| │   └─AdaptiveMaxPool2d: 2-10        |                 |
| │   └─AdaptiveMaxPool2d: 2-11        |                 |
| │   └─AdaptiveMaxPool2d: 2-12        |                 |
| ├─Sequential: 1-3                    |                 |
| │   └─Linear: 2-13                   | 22,283,264      |
| │   └─ReLU: 2-14                     |                 |
| │   └─Linear: 2-15                   | 2,050           |
|---------------------------------------|-----------------|
| Total params:                         | 22,656,130      |
| Trainable params:                     | 22,656,130      |
| Non-trainable params:                 | 0               |



# For NAS search space
We explore the following search spaces for all three components:
## Feature Engineering: We define the search space for the filter
Size of the first convolutional layer as ranging from 1 to 9 (1,
3, 5, 7, 9).
## SPP Layer: We experiment with five different filter sizes for
The first SPP (Spatial Pyramid Pooling) layer, spanning from
1 to 5 (1, 2, 3, 4, 5).
## Fully-Connected Layers: We customize the feature size for
Two fully-connected layers within the following ranges: 128,
256, 512, 1024, 2048, 4096, and 8192.
