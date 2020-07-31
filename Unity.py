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
resize600 = transforms.Resize(500)
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
 ])    
    
    def Unity_function(im):
  transform = transforms.Compose([
      transforms.Resize(600),
      transforms.ToTensor()                            
  ])
  to_pil = transforms.ToPILImage()
  im_tensor = transform(im)
  
!mkdir webims
!curl -o ./webims/flower.jpg https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500

im_flower = Image.open('./webims/flower.jpg')
resize600= transforms.Resize(500)
im_flower = resize600(im_flower) #Scale longest dimension down to 500 pixels
to_tensor= transforms.ToTensor()
im_puppy_tensor = to_tensor(grayscale(im_flower))


display(to_pil(im_flower_tensor))
display(to_pil(rescale(im_flower_tensor,perc1=40,perc2=60)))

#to_pil = transforms.ToPILImage()

#vertical_kernel = torch.Tensor([[-1, 0, 1],
                               # [-2, 0, 2],
                         #       [-1, 0, 1]])
#im_flower_vertical_tensor = apply_convolution(im_puppy_tensor, vertical_kernel)
#im_flower_vertical_tensor = rescale(im_puppy_vertical_tensor, perc1=20, perc2=98)
#display(to_pil(im_flower_vertical_tensor))

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
    img = Image.open('./webims/flower.jpg')

    transformed = Unity_transform(img)

    display(img)

#test()
