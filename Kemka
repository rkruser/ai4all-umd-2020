# Load a random image from the web
!mkdir webims
!curl -o ./webims/duck.jpg https://upload.wikimedia.org/wikipedia/commons/b/b6/Duckling_-_domestic_duck.jpg

# Load some necessary library functions from the PIL library and torchvision (a module in torch)
from torchvision import transforms
from PIL import Image

duck = Image.open('webims/duck.jpg')
transform_to_tensor = transforms.ToTensor()
duck_tensor = transform_to_tensor(duck)
print("Image dimensions:", duck_tensor.size())
display(duck)

duck2 = duck_tensor[:,10:500,500:1067]
to_pil = transforms.ToPILImage() # Function to convert from tensor back to PIL
display(to_pil(duck2))

transform_image = transforms.Compose([
    transforms.Resize((100,100)), # Resize image to 100 by 100
    transforms.Grayscale(), # Convert to black and white
    transforms.ToTensor() # Convert resulting PIL image to tensor
])
duck_tensor_2 = transform_image(duck)
print(duck_tensor_2.size())

to_pil = transforms.ToPILImage() # Function to convert from tensor back to PIL
display(to_pil(duck_tensor_2)) # view the transformed image
