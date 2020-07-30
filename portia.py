import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import random

def apply_convolution(im_tensor, kernel, color):
  if color:
    conv = nn.Conv2d(3,1,kernel.size()[1:], bias = False)
    conv.weight.data = kernel.unsqueeze(0)
  else:
    conv = nn.Conv2d(1,1,kernel.size(), bias = False)
    conv.weight.data = kernel.unsqueeze(0).unsqueeze(0)
  return conv(im_tensor.unsqueeze(0)).squeeze(0)

def portiaImP(im):
  to_pil = transforms.ToPILImage()
  to_tensor = transforms.ToTensor()
  transform_image_small = transforms.Compose([
      transforms.Resize((45,50)), # Resize image to 45 by 45
      transforms.Grayscale(),
      transforms.ToTensor() # Convert resulting PIL image to tensor
  ])

  transform_image_large = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor() # Convert resulting PIL image to tensor
  ])

  kernel = torch.Tensor([ [0, -1, 0] ,
                          [-1, 3, -1],
                          [0, -1, 0] 
                            ])

  # Make the image smaller
  imSmallTensor = transform_image_small(im)
  imBig = to_pil(transform_image_large(im))
  imSmallTensor = (apply_convolution(imSmallTensor, kernel, False))

  for i in range(35):
    number1 = random.randint(0,224)
    number2 = random.randint(0,224)
    # Place the cat image onto the base image
    imBig.paste(to_pil(imSmallTensor),(number1, number2))

  display(imBig)

!mkdir webims
!curl -o ./webims/image.jpg https://www.peta.org/wp-content/uploads/2014/01/goldfish-sxc.hu_.jpg
im = Image.open('./webims/image.jpg')
portiaImP(im)
