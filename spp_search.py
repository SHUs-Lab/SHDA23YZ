#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn.functional as F
import object_detection

import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper

@model_wrapper      
# this decorator should be put on the out most
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


@model_wrapper
class ModelSpace(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SPPNet, self).__init__()
        filter_size1 = nn.ValueChoice([1,3,5,7,9])
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=filter_size1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        filter_size2 = nn.ValueChoice([1,2,3,4,5])
        self.spp_layer = nn.ModuleList([
            nn.AdaptiveMaxPool2d((filter_size2, filter_size2)),  # SPP level 1
            nn.AdaptiveMaxPool2d((2, 2)),  # SPP level 2
            nn.AdaptiveMaxPool2d((1, 1))   # SPP level 3
        ])
        feature = nn.ValueChoice([128,256, 512, 1024, 2048, 4096, 8192])

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * (1 + 4 + 16 + filter_size2*filter_size2), feature),
            nn.ReLU(inplace=True),
            nn.Linear(feature, num_classes)
        )
                
    def forward(self, x):
        x = self.conv_layers(x)
        spp_outputs = [layer(x) for layer in self.spp_layer]
        spp_outputs = [output.view(output.size(0), -1) for output in spp_outputs]
        spp_features = torch.cat(spp_outputs, dim=1)
        return self.fc_layers(spp_features)    
    

model_space = ModelSpace()
model_space




import nni.retiarii.strategy as strategy
search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted




import nni

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import object_detection


def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    model = model_cls()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    for epoch in range(3):
        accuracy = object_detection.train_tuning(model)


        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

        # report final test result
    nni.report_final_result(accuracy)




from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'sppnet_search'




exp_config.max_trial_number = 4   # spawn 4 trials at most
exp_config.trial_concurrency = 2  # will run two trials concurrently




exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True




exp.run(exp_config, 8081)




import os
from pathlib import Path


def evaluate_model_with_visualization(model_cls):
    model = model_cls()
    # dump the model into an onnx
    if 'NNI_OUTPUT_DIR' in os.environ:
        dummy_input = torch.zeros(1, 3, 32, 32)
        torch.onnx.export(model, (dummy_input, ),
                          Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')
    evaluate_model(model_cls)

