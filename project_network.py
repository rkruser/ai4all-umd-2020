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
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 2)
        self.pool = nn.MaxPool2d(2, 1)

        self.conv2 = nn.Conv2d(64, 128, 5, 2)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 5, 2)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, 5, 2)

        self.linLay1 = nn.Linear(512*224*224, 1024)
        self.linLay2 = nn.Linear(1024, 512)
        self.linLay3 = nn.Linear(512, 256)
        self.linLay4 = nn.Linear(256, 185)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 512*224*224)
        x = F.relu(self.linLay1(x))
        x = F.relu(self.linLay2(x))
        x = F.relu(self.linLay3(x))
        x = self.linLay4(x)
        return x
