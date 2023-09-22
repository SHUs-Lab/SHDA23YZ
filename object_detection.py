# Download TorchVision repo to use some files from
# references/detection
get_ipython().system('pip install pycocotools --quiet')
get_ipython().system('git clone https://github.com/pytorch/vision.git')
get_ipython().system('git checkout v0.3.0')

get_ipython().system('cp vision/references/detection/utils.py ./')
get_ipython().system('cp vision/references/detection/transforms.py ./')
get_ipython().system('cp vision/references/detection/coco_eval.py ./')
get_ipython().system('cp vision/references/detection/engine.py ./')
get_ipython().system('cp vision/references/detection/coco_utils.py ./')



# Basic python and ML Libraries
import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils
import transforms as T
get_ipython().system('pip install -q torch_snippets lovely-tensors torchinfo')

from torch_snippets import *
from IPython import display 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
display.set_matplotlib_formats('svg')

from torchinfo import summary

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from xml.etree import ElementTree as et
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# The data is not big. It contains $1600$ images in the train folder and $400$ images in the test folder.  
root = './train/'

# we have two labels
labels = ['culvert']
label2targets = {l: t for t, l in enumerate(labels)}
targets2label = {t: l for l, t in label2targets.items()}
num_classes = len(targets2label)



def plot_img_bbox(img, target):
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for box in target: 
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        a.add_patch(rect)
    plt.show()


def preprocess_img(img):
    img = torch.tensor(img).permute(2, 0 ,1)
    return img.float()

class ImageDataset(Dataset):
    def __init__(self, root=root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_paths = sorted(Glob(self.root + '/*.tif'))
        self.xlm_paths = self.root + '1*.xml'
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        w, h = 100, 100
        img_path = self.img_paths[idx]
        xlm_path = self.xlm_paths
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        img = np.array(img.resize((w, h), 
                       resample=Image.Resampling.BILINEAR))/255.
        xlm = et.parse(xlm_path)
        objects = xlm.findall('object')
        labels = []
        boxes = []
        for obj in objects:
            label = obj.find('name').text
            labels.append(label)
            XMin = float(obj.find('bndbox').find('xmin').text)
            YMin = float(obj.find('bndbox').find('ymin').text)
            XMax = float(obj.find('bndbox').find('xmax').text)
            YMax = float(obj.find('bndbox').find('ymax').text)
            bbox = [XMin / W, YMin / H, XMax / W, YMax / H]
            bbox = (bbox * np.array([w, h, w, h])).astype(np.int16).tolist()
            boxes.append(bbox)
        target = {}
        target['labels'] = torch.tensor([label2targets[label] for label in labels]).long()
        target['boxes'] = torch.tensor(boxes).float()
        img = preprocess_img(img)
        return img, target
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))

val_root = './test/'
    
train_ds = ImageDataset()
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=train_ds.collate_fn)

val_ds = ImageDataset(root=val_root)
val_dl = DataLoader(val_ds, batch_size=2, shuffle=True, collate_fn=val_ds.collate_fn)

img, target = train_ds[10]
plot_img_bbox(img.permute(1,2,0), target['boxes'])







# SPPNet implementation
class SPPNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SPPNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.spp_layer = nn.ModuleList([
            nn.AdaptiveMaxPool2d((4, 4)),  # SPP level 1
            nn.AdaptiveMaxPool2d((2, 2)),  # SPP level 2
            nn.AdaptiveMaxPool2d((1, 1))   # SPP level 3
        ])
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * (1 + 4 + 16 + 64), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        spp_outputs = [layer(x) for layer in self.spp_layer]
        spp_outputs = [output.view(output.size(0), -1) for output in spp_outputs]
        spp_features = torch.cat(spp_outputs, dim=1)
        return self.fc_layers(spp_features)




from torchinfo import summary




# test the model
model = SPPNet(in_channels=3, num_classes=1)
# model.to(device)
summary(model)


# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)



# training for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, train_dl, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, val_dl, device=device)



# training for one epoch
def train_tuning(model):
    # training for one epoch
    train_one_epoch(model, optimizer, train_dl, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    accuracy = evaluate(model, val_dl, device=device)
return accuracy

