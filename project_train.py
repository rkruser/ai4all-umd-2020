import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils.data import Dataloader
import torchvision.transforms as transforms
import os
import pickle

from data_loader import LeafSnapLoader


def train(net, num_epochs=10, save_location='../drive/My\ Drive/models'):
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    leafsnap_train_dataset = LeafSnapLoader(mode='train')
    leafsnap_val_dataset = LeafSnapLoader(mode='val')
    leafsnap_test_dataset = LeafSnapLoader(mode='test')

    leafsnap_train_loader = Dataloader(leafsnap_train_dataset, batch_size=64, shuffle=True)
    leafsnap_validate_loader = Dataloader(leafsnap_val_dataset, batch_size=64, shuffle=False)
    leafsnap_test_loader = Dataloader(leafsnap_test_dataset, batch_size=64, shuffle=False)

    # Write the training function. Use GPUs if available
    # Periodically save the trained model in the save location
    # Every few epochs, run the model on the validation set
    # At the end of training, run the model on the test set


