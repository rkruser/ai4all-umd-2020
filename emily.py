import torch # general PyTorch
from torchvision import transforms # Image transforming library
import torch.nn as nn # Neural network library
from PIL import Image # Image class from python imaging library (PIL)
import numpy as np

def apply_convolution(im_tensor, kernel, bias=None, stride=1, padding=0, color=False):
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

def emily_function(im):
  transform = transforms.Compose([
      transforms.Resize(600),
      transforms.ToTensor()                            
  ])
  to_pil = transforms.ToPILImage()
  im_tensor = transform(im)

  vertical_kernel = torch.Tensor([[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]],[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]],[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]])

  im_vertical_tensor = apply_convolution(im_tensor, vertical_kernel,color=True)
  im_vertical_tensor = im_vertical_tensor.clamp(0,1)

  blur_kernel = torch.ones(10,10)
  im_blur = apply_convolution(im_vertical_tensor, blur_kernel)
  im_blur = rescale(im_blur, perc1=0, perc2=100)

  sliced_image=im_blur[:,:,:]
  zeros = torch.zeros(sliced_image.size()[1:])
  red_slice = sliced_image[0,:,:]
  red_slice = torch.stack([red_slice, zeros, zeros])
  #display(to_pil(red_slice))
  return to_pil(red_slice)

#!mkdir webims
#!curl -o ./webims/image.jpg https://static01.nyt.com/images/2020/04/22/arts/22kimmelman-skyscrapers1/22kimmelman-skyscrapers1-videoSixteenByNineJumbo1600.jpg 
#im = Image.open('./webims/image.jpg')
#emily_function(im)
