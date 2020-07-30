# Importing libraries
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


#resnet18 = models.resnet18(pretrained=False, num_classes=185) # A pre-defined neural network model to compare to

# round up: (input_size - kernel_size + 1 + 2*padding)/stride
# Fill in your own convolutional neural network
# Should takes batches of 3x224x224 LeafSnap images and return batches of vectors of size 185 (the number of classifications in leafsnap)
# Use the modules nn.Conv2d, nn.Linear, nn.ReLU, and other pytorch functions you deem necessary
class YourNetwork(nn.Module):
   def __init__(self):
        super(YourNetwork, self).__init__()# Ask about
         
        # Apply 4 convultions and 3 max pools
        self.conv1 = nn.Conv2d(3, 64, 5, 2) # Input channel, output channel, kernel size, stride
        # First max pool
        self.pool = nn.MaxPool2d(2, 2) # kernel size, stride

        self.conv2 = nn.Conv2d(64, 128, 5, 2)# Input channel, output channel, kernel size, stride
        # Second max pool
        #self.pool2 = nn.MaxPool2d(2, 2) # kernel size, stride
        self.conv3 = nn.Conv2d(128, 256, 3, 1)# Input channel, output channel, kernel size, stride  
        self.conv4 = nn.Conv2d(256, 512, 3, 2)# Input channel, output channel, kernel size, stride

        # lower( ( size + 2*padding - kernel)/stride + 1 )
      
        # lower((224-5)/2+1 ) = 110 (conv1)
        # 55 (pool)
        # lower((55-5)/2+1) = 26 (conv2)
        # 13 (pool)
        # conv3 + pool: lower((13-3+1)/2) = 5
        # conv4: lower((5-3)/2+1) = 2       

        # Goal: 512 x 2 x 2 = 2048

        # Apply 4 linear layers
        self.linLay1 = nn.Linear(2048, 1024)# Input size(channels*pixels), output size
        self.linLay2 = nn.Linear(1024, 512)# Input size, output size
        self.linLay3 = nn.Linear(512, 256)# Input size, output size
        self.linLay4 = nn.Linear(256, 185)# Input size, output size

   def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))         # Call the max pools in terms of the applied convolutions
        #  Conv1: batch_size x 64 x 110 x 110
        #  Pool: (110-2+1)/1 = batch_size x 64 x 109 x 109
        x = self.pool(F.relu(self.conv2(x)))
        #  Conv2: (109-5+1)/2 = 52.5 --> 53   batch_size x 128 x 53 x 53
        #  Pool2: (53-2+1)/2 = 26    batch_size x 128 x 26 x 26
        x = self.pool(F.relu(self.conv3(x)))
        # Conv3: batch_size x 256 x 11 x 11
        # Pool2: batch_size x 256 x 5 x 5
        x = F.relu(self.conv4(x))
        # Conv4: batch_size x 512 x 1 x 1
        # Pool2: Can't apply

        #print(x.size())
         
        # Resize the tensor so that the linear layers can use it
        x = x.view(-1, 512*2*2) # Change
         
        # Apply the reLus to the linLays
        x = F.relu(self.linLay1(x))
        x = F.relu(self.linLay2(x))
        x = F.relu(self.linLay3(x))
        x = self.linLay4(x)
      
        # Return the final tensor
        return x
