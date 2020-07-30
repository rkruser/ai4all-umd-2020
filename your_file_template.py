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

# To run the test in colaboratory, uncomment test(), then go to a code cell and type "run {your file name}.py"
# Or comment out test(), and in a code cell run: "from {your file name without .py extension} import test" and "test()"
