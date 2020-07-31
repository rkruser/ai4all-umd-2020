import torch
from torchvision import transforms
from PIL import Image
#!mkdir webims
#!curl -o ./webims/labrador.jpg https://thumbs-prod.si-cdn.com/ej9KRK9frB5AXD6W9LXKFnuRc-0=/fit-in/1600x0/https://public-media.si-cdn.com/filer/ad/7b/ad7b3860-ad5f-43dc-800e-af57830cd1d3/labrador.jpg
#pic = Image.open('webims/labrador.jpg')

def shubham_function(img):
  transformer = transforms.ToTensor()
  #transform_image = transforms.Compose([transforms.Resize((30,50)),transforms.ToTensor()])
#dog_tensor = transformer(pic)
#display(pic)
#dogpic = dog_tensor[:,10:420,200:800]
  to_pil = transforms.ToPILImage() 
  im_tensor_new = transformer(img)
  height = im_tensor_new.size(1)
  width = im_tensor_new.size(2)
  im_tensor_new = im_tensor_new[:,int(0.2*height):int(0.8*height),int(0.2*width):int(0.8*width)]
  
  return to_pil(im_tensor_new) 
#display(to_pil(dogpic))


#dog_tensor_new = transform_image(pic)
#to_pil = transforms.ToPILImage() 
#display(to_pil(dog_tensor_new)) 
