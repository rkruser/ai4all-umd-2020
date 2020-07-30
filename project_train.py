#importing libraries, functions, etc.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
Dataloader = DataLoader
import torchvision.transforms as transforms
import os
import sys
import pickle
import matplotlib.pyplot as plt
from data_loader import LeafSnapLoader, leafsnap_collate_fn

default_im_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]) #resize image and make it into a tensor
augmented_im_transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor()]) 

criterion = nn.CrossEntropyLoss()

#defining our training function
def train(net, device='cpu', start_epoch=0, optimizer=None, num_epochs=10, save_location='./', im_transform = default_im_transform):
    leafsnap_train_dataset = LeafSnapLoader(mode='train', transform=im_transform) #data is used for training purpose
    leafsnap_val_dataset = LeafSnapLoader(mode='val', transform=default_im_transform) #data is used for validating purpose
    
    print("Train dataset length", len(leafsnap_train_dataset))
    print("Validation dataset length", len(leafsnap_val_dataset))

    #taking the data sets and giving them a batch size of 64
    leafsnap_train_loader = Dataloader(leafsnap_train_dataset, batch_size=64, shuffle=True, collate_fn=leafsnap_collate_fn) 
    leafsnap_validate_loader = Dataloader(leafsnap_val_dataset, batch_size=64, shuffle=False, collate_fn=leafsnap_collate_fn)

    # Write the training function. Use GPUs if available

#    net = net.to(device) 

    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=0.0002)


    average_train_losses = []
    validation_accuracies = []
    for epoch in range(start_epoch,num_epochs):  # loop over the dataset multiple times

        net.train()
        epoch_losses = []
        running_loss = 0.0
        for i, data in enumerate(leafsnap_train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels =  data['image'].to(device), data['species_index'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs) # outputs have size batch_size x 185
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_losses.append(loss.item()) # Save the losses in each batch
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i + 1, running_loss / 10))
                running_loss = 0.0

        average_train_losses.append(torch.Tensor(epoch_losses).mean().item())

        # Every few epochs, run the model on the validation set
        if epoch % 1 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                net.eval()
                for data in leafsnap_validate_loader:
                    images, labels = data['image'].to(device), data['species_index'].to(device)
                    outputs = net(images) # batch_size x 185
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
              # return to this and compute accuracy

            print('Accuracy of the network on the validation set: %d %%' % (
                    100 * correct / total))

            validation_accuracies.append((epoch,correct/total))


        # Periodically save the trained model in the save location
        if (epoch % 5 == 4) or (epoch==num_epochs-1): 
            torch.save((net.state_dict(),optimizer.state_dict()), os.path.join(save_location,'model_{0}.pth'.format(epoch)))
            pickle.dump((average_train_losses, validation_accuracies), open(os.path.join(save_location,'stats_{0}.pkl'.format(epoch)),'wb'))


    print('Finished Training')
    
    
def test(net, device='cpu'):
    # Load test dataset
    net = net.to(device)
    net.eval()
    leafsnap_test_dataset = LeafSnapLoader(mode='test',transform=default_im_transform) #data is used for testing purpose
    print("Test set size", len(leafsnap_test_dataset))

    leafsnap_test_loader = Dataloader(leafsnap_test_dataset, batch_size=64, shuffle=False, collate_fn=leafsnap_collate_fn)

    # Run trained network on test data and compute accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(leafsnap_test_loader):
            print("Test batch {0} of {1}".format(i, len(leafsnap_test_loader)))
            images, labels = data['image'].to(device), data['species_index'].to(device)
            outputs = net(images) # batch_size x 185
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--nepochs', type=int, default=50)
    parser.add_argument('--save_location', type=str, default='./models')
    parser.add_argument('--view_stats', action='store_true')
    parser.add_argument('--stats_file', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    opt = parser.parse_args()

    print("Options selected:", opt)

    if opt.view_stats:
        stats = pickle.load(open(opt.stats_file,'rb'))
        print(stats)
        sys.exit()

    device = torch.device("cuda:{}".format(opt.gpuid) if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    from project_network import *
    if opt.resnet:
        net = models.resnet18(pretrained=False, num_classes=185)
    else:
        net = YourNetwork()

    net = net.to(device)

    if opt.load_from is not None:
        net_params, optim_params = torch.load(opt.load_from, map_location=device)
        net.load_state_dict(net_params)
        optimizer = optim.Adam(net.parameters(), lr=0.0002)
        optimizer.load_state_dict(optim_params)
    else:
        optimizer = optim.Adam(net.parameters(), lr=0.0002)

    if opt.augment:
        tform = augmented_im_transform
    else:
        tform = default_im_transform

    if opt.train:
        if not os.path.isdir(opt.save_location):
            os.makedirs(opt.save_location)
        train(net, start_epoch=opt.start_epoch, device=device, optimizer=optimizer, num_epochs=opt.nepochs, save_location=opt.save_location,
               im_transform = tform)
    else:
        test(net, device=device)




