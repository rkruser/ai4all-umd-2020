import torch # general PyTorch
from torchvision import transforms # Image transforming library
import torch.nn as nn # Neural network library
from PIL import Image # Image class from python imaging library (PIL)
import numpy as np


# Print a list of tensors
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

  !mkdir webims
!curl -o ./webims/puppy.jpg https://i.pinimg.com/474x/57/92/6a/57926a0d9ac21aa58e03e018087a21bb--german-shepherd-pups-shepherd-dogs.jpg

puppy = Image.open('./webims/puppy.png')
puppy = resize600(puppy) #Scale longest dimension down to 600 pixels
puppy_tensor = to_tensor(grayscale(puppy))

display(to_pil(puppy_tensor))
display(to_pil(rescale(puppy_tensor,perc1=40,perc2=60)))

vertical_kernel = torch.Tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
puppy_vertical_tensor = apply_convolution(puppy_tensor, vertical_kernel)
puppy_vertical_tensor = rescale(puppy_vertical_tensor, perc1=20, perc2=98)
display(to_pil(puppy_vertical_tensor))

horizontal_kernel = torch.Tensor([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]])
puppy_horizontal_tensor = apply_convolution(puppy_tensor, horizontal_kernel)
puppy_horizontal_tensor = rescale(puppy_horizontal_tensor, perc1=20, perc2=98)
display(to_pil(puppy_horizontal_tensor))

blur_kernel = torch.ones(10,10)
puppy_blur = apply_convolution(puppy_tensor, blur_kernel)
puppy_blur = rescale(puppy_blur, perc1=0, perc2=100)
display(to_pil(puppy_blur))
