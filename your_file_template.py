import torch
from torchvision import transforms
from PIL import Image

# Define any helper functions such as convolutions here



# Take a PIL Image, transform it, and return the transformed PIL image
def your_name_transform(img):
    tensor_transform = transforms.Compose([
        ...
        ...
        ])

    im_tensor = tensor_transform(img)

    # perform operations on image

    to_pil = transforms.ToPILImage()

    return to_pil(im_tensor)


def test():
    img = Image.open('/path/to/image.jpg')

    transformed = your_name_transform(img)

    display(img)

#test()

