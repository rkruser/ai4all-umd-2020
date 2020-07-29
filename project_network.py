# Importing libraries
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
        super(Net, self).__init__()# Ask about
         
        # Apply 4 convultions and 3 max pools
        self.conv1 = nn.Conv2d(3, 64, 5, 2) # Input channel, output channel, kernel size, stride
        # First max pool
        self.pool = nn.MaxPool2d(2, 1) # kernel size, stride

        self.conv2 = nn.Conv2d(64, 128, 5, 2)# Input channel, output channel, kernel size, stride
        # Second max pool
        self.pool2 = nn.MaxPool2d(2, 2) # kernel size, stride
        self.conv3 = nn.Conv2d(128, 256, 5, 2)# Input channel, output channel, kernel size, stride  
        self.conv4 = nn.Conv2d(256, 512, 5, 2)# Input channel, output channel, kernel size, stride

        # Apply 4 linear layers
        self.linLay1 = nn.Linear(512*224*224, 1024)# Input size(channels*pixels), output size
        self.linLay2 = nn.Linear(1024, 512)# Input size, output size
        self.linLay3 = nn.Linear(512, 256)# Input size, output size
        self.linLay4 = nn.Linear(256, 185)# Input size, output size

    def forward(self, x):
        # Call the max pools in terms of the applied convolutions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))
         
        # Resize the tensor so that the linear layers can use it
        x = x.view(-1, 512*224*224)
         
        # Apply the reLus to the linLays
        x = F.relu(self.linLay1(x))
        x = F.relu(self.linLay2(x))
        x = F.relu(self.linLay3(x))
        x = self.linLay4(x)
      
        # Return the final tensor
        return x
