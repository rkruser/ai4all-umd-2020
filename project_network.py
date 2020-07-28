import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


resnet18 = models.resnet18(pretrained=False, num_classes=185) # A pre-defined neural network model to compare to


# Fill in your own convolutional neural network
# Should takes batches of 3x224x224 LeafSnap images and return batches of vectors of size 185 (the number of classifications in leafsnap)
# Use the modules nn.Conv2d, nn.Linear, nn.ReLU, and other pytorch functions you deem necessary
class YourNetwork(nn.Module):
    def __init__(self):
        pass



    def forward(self, x):
        pass
