import torch # general PyTorch
from torchvision import transforms # Image transforming library
import torch.nn as nn # Neural network library
from PIL import Image # Image class from python imaging library (PIL)
import numpy as np


def print_tensors(tensor_list):
  for t in tensor_list:
    print(t.size(), '\n', t)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
resize600 = transforms.Resize(600)
grayscale = transforms.Grayscale()

def apply_convolution(im_tensor, kernel, bias=None, stride=1, padding=0):
  if len(kernel.size()) == 2:
    kernel = kernel.unsqueeze(0).unsqueeze(0)
  elif len(kernel.size()) == 3:
    kernel = kernel.unsqueeze(0)
  return nn.functional.conv2d(im_tensor.unsqueeze(0), kernel, bias=bias, stride=stride, padding=padding).squeeze(0)

def rescale(im_tensor, perc1 = 20, perc2 = 98):
  numpy_tensor = im_tensor.numpy()
  pc1 = np.percentile(numpy_tensor, perc1)
  pc2 = np.percentile(numpy_tensor, perc2)
  return ((im_tensor-pc1)/(pc2-pc1)).clamp(0,1)
    
def Anu_function(im):
  transform = transforms.Compose([
      transforms.Resize(600),
      transforms.Grayscale(),
      transforms.ToTensor()                            
  ])
  to_pil = transforms.ToPILImage()
  im_tensor = transform(im)
  im_tensor = rescale(im_tensor,perc1=40,perc2=60)
  
  return to_pil(im_tensor)
  
#!mkdir webims
#!curl -o ./webims/puppy.jpg https://i.pinimg.com/474x/57/92/6a/57926a0d9ac21aa58e03e018087a21bb--german-shepherd-pups-shepherd-dogs.jpg

#im_puppy = Image.open('./webims/puppy.jpg')
#resize600= transforms.Resize(600)
#im_puppy = resize600(im_puppy) #Scale longest dimension down to 600 pixels
#to_tensor= transforms.ToTensor()
#im_puppy_tensor = to_tensor(grayscale(im_puppy))


#display(to_pil(im_puppy_tensor))
#display(to_pil(rescale(im_puppy_tensor,perc1=40,perc2=60)))

#to_pil = transforms.ToPILImage()

#vertical_kernel = torch.Tensor([[-1, 0, 1],
                               # [-2, 0, 2],
                         #       [-1, 0, 1]])
#im_puppy_vertical_tensor = apply_convolution(im_puppy_tensor, vertical_kernel)
#im_puppy_vertical_tensor = rescale(im_puppy_vertical_tensor, perc1=20, perc2=98)
#display(to_pil(im_puppy_vertical_tensor))

#return to_pil(im_tensor)

#horizontal_kernel = torch.Tensor([[1, 2, 1],
                                #  [0, 0, 0],
                                #  [-1, -2, -1]])
#im_puppy_horizontal_tensor = apply_convolution(im_puppy_tensor, horizontal_kernel)
#im_puppy_horizontal_tensor = rescale(im_puppy_horizontal_tensor, perc1=20, perc2=98)
#display(to_pil(im_puppy_horizontal_tensor))

#blur_kernel = torch.ones(10,10)
#im_puppy_blur = apply_convolution(im_puppy_tensor, blur_kernel)
#im_puppy_blur = rescale(im_puppy_blur, perc1=0, perc2=100)
#display(to_pil(im_puppy_blur))

def test():
    img = Image.open('./webims/puppy.jpg')

    transformed = Anu_transform(img)

    display(img)

#test()
