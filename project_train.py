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

im_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]) #resize image and make it into a tensor
criterion = nn.CrossEntropyLoss()

#defining our training function
def train(net, num_epochs=10, save_location='./'):
    leafsnap_train_dataset = LeafSnapLoader(mode='train', transform=im_transform) #data is used for training purpose
    leafsnap_val_dataset = LeafSnapLoader(mode='val', transform=im_transform) #data is used for validating purpose
    
    print("Train dataset length", len(leafsnap_train_dataset))
    print("Validation dataset length", len(leafsnap_val_dataset))

    #taking the data sets and giving them a batch size of 64
    leafsnap_train_loader = Dataloader(leafsnap_train_dataset, batch_size=64, shuffle=True, collate_fn=leafsnap_collate_fn) 
    leafsnap_validate_loader = Dataloader(leafsnap_val_dataset, batch_size=64, shuffle=False, collate_fn=leafsnap_collate_fn)


    # Write the training function. Use GPUs if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    net = net.to(device) 

    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    average_train_losses = []
    validation_accuracies = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

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
            if i % 5 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        average_train_losses.append(torch.Tensor(epoch_losses).mean())

        # Every few epochs, run the model on the validation set
        if epoch % 1 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
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
        if epoch % 5 == 0:
            torch.save(net.state_dict(), os.path.join(save_location,'model_{0}.pth'.format(epoch)))
            pickle.dump((average_train_losses, validation_accuracies), open(os.path.join(save_location,'stats.pkl'),'wb'))


    print('Finished Training')
    
    
def test(net):
    # Load test dataset
    leafsnap_test_dataset = LeafSnapLoader(mode='test',transform=im_transform) #data is used for testing purpose
    leafsnap_test_loader = Dataloader(leafsnap_test_dataset, batch_size=64, shuffle=False, collate_fn=leafsnap_collate_fn)

    # Run trained network on test data and compute accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in leafsnap_test_loader:
            images, labels = data['image'].to(device), data['species_index'].to(device)
            outputs = net(images) # batch_size x 185
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    from project_network import YourNetwork
    net = YourNetwork()

    train(net, num_epochs=50, save_location='./models')




