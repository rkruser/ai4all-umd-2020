# Run resnet50 pretrained on imagenet, your neural network model,
#  and all student image transformations
#  Collect results in a dictionary
import traceback
from flask import Flask, jsonify, request, render_template
from flask_ngrok import run_with_ngrok

#from utils import read_file, transform_image, get_topk, model  #render_prediction

app = Flask(__name__)

import os
import json
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

from PIL import Image
import requests
from io import BytesIO
import base64

from data_loader import ClassLoader

# Student image transforms
# from Anu import ...
# from shubham import ...
# from emily import ...
# from portia import ...
# from kemka import ...
# from unity import ...


def read_file(upload=None, url=None):
    if (upload is not None) and upload.filename:
        in_memory_file = BytesIO()
        upload.save(in_memory_file)
        img = Image.open(in_memory_file)
        return img

    elif url is not None:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    else:
        raise NameError('Invalid file/url')

def to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('ascii')

# Transform input into the form our model expects
def transform_image(pil_image):
    input_transforms = [
        transforms.Resize(255),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225]
        )
    ]
    my_transforms = transforms.Compose(input_transforms)
    timg = my_transforms(pil_image)                       # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg

leafsnap_transform_image = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor()
])

def get_topk(model, input_tensor, k=5):
    outputs = model(input_tensor)                 # Get likelihoods for all ImageNet classes
    values, indices = torch.topk(outputs, k)              # Extract top k most likely classes
    values = values.data.cpu().numpy()[0]
    indices = indices.data.cpu().numpy()[0]
    return values, indices

resnet50_imagenet_model = models.resnet50(pretrained=True)
resnet50_imagenet_model.eval()
img_class_map = None
mapping_file_path = 'index_to_name.json'                  # Human-readable names for Imagenet classes
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)

device = 'cpu' #We don't need to bother with the GPU when testing the trained model on small sets of images
your_net = pnet.YourNetwork() # A special argument I added to your network --Ryen
net_weights, _ = torch.load('./models/your_model/your_model_49.pth',map_location=device) # The second return value is the optimizer weights, which we don't need now
your_net.load_state_dict(net_weights)
your_net.eval()
leaf_species_name_mapping = ClassLoader()
downsize = transforms.Resize(30)

# Need to return a list of dictionaries with keys 'model', 'label', 'score', 'image'
# Ryen will write this function and get back to you
def collect_outputs(input_pil_image):
  resnet50_im = transform_image(input_pil_image)
  your_net_im = leafsnap_transform_image(input_pil_image).unsqueeze(0)

  r50_vals, r50_inds = get_topk(resnet50_imagenet_model, resnet50_im, 5)
  your_vals, your_inds = get_topk(your_net, your_net_im, 5)

  image_net_results = []
  for value, idx in zip(r50_vals, r50_inds):
    image_net_results.append({
        "model": "ImageNet Resnet50 Pretrained",
        "category": img_class_map.get(str(idx), "Unknown")[1],
        "score": str(value),
        "image": None
    })

  your_net_results = []
  for value, idx in zip(your_vals, your_inds):
    species_name = leaf_species_name_mapping.ind2str(idx)
    species_dir = os.path.join('./data/leafsnap/dataset/images/field/',species_name)
    species_file = os.listdir(species_dir)[0]
    your_net_results.append({
        "model": "Your Network Trained on Leafsnap",
        "category": species_name,
        "score": str(value),
        "image": None #to_base64(downsize(Image.open(species_file)))
    })

  # In place of "None", write
  #  to_base64( student_transform( img ) )
  #  student_transform must take a PIL image and return a PIL image
  #  The returned image must be no larger than 256 by 256, preferably smaller
  student_image_transforms = [
    {
      "model": "Shubham",
      "category": '-',
      "score": '-',
      "image": None 
    },
    {
      "model": "Portia",
      "category": '-',
      "score": '-',
      "image": None # Fill this in    
    },
    {
      "model": "Kemka",
      "category": '-',
      "score": '-',
      "image": None # Fill this in    
    },
    {
      "model": "Emily",
      "category": '-',
      "score": '-',
      "image": None # Fill this in    
    },
    {
      "model": "Unity",
      "category": '-',
      "score": '-',
      "image": None # Fill this in    
    },
    {
      "model": "Anu",
      "category": '-',
      "score": '-',
      "image": None # Fill this in    
    },
  ]

  all_results = image_net_results + your_net_results + student_image_transforms

  return all_results
  