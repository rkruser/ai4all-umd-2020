#importing libraries, functions, etc.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils.data import Dataloader
import torchvision.transforms as transforms
import os
import pickle
import matplotlib.pyplot as plt
from data_loader import LeafSnapLoader

#defining the function to show the images later on in code when testing
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#defining our training function
def train(net, num_epochs=10, save_location='../drive/My\ Drive/models'):
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]) #resize image and make it into a tensor
    leafsnap_train_dataset = LeafSnapLoader(mode='train') #data is used for training purpose
    leafsnap_val_dataset = LeafSnapLoader(mode='val') #data is used for validating purpose
    leafsnap_test_dataset = LeafSnapLoader(mode='test') #data is used for testing purpose

    #taking the data sets and giving them a batch size of 64
    leafsnap_train_loader = Dataloader(leafsnap_train_dataset, batch_size=64, shuffle=True) 
    leafsnap_validate_loader = Dataloader(leafsnap_val_dataset, batch_size=64, shuffle=False)
    leafsnap_test_loader = Dataloader(leafsnap_test_dataset, batch_size=64, shuffle=False)

    # Write the training function. Use GPUs if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    inputs, labels = data['image'].to(device), data['species_index'].to(device)

    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(leafsnap_train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels =  data['image'].to(device), data['species_index'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # Every few epochs, run the model on the validation set
        if epoch % 3 ==0:
          for i, data in enumerate(leafsnap_validate_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels =  data['image'].to(device), data['species_index'].to(device)

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # Periodically save the trained model in the save location
        if epoch % 5 == 0:
          save_location = '../drive/My\ Drive/models'
          torch.save(net.state_dict(), save_location)
    print('Finished Training')
    
    
# At the end of training, run the model on the test set
dataiter = iter(leafsnap_test_loader)  #going through the data
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images)) # print images from test set
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.load_state_dict(torch.load(save_location)) #extracting out saved model

outputs = net(images) #what the NN thinks the images are

#index of highest energy (how much the network thinks an image belongs to the class)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                          for j in range(4)))

#Accuracy of network as a whole
correct = 0
total = 0
with torch.no_grad():
    for data in leafsnap_test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#Accuracy by class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in leafsnap_test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

