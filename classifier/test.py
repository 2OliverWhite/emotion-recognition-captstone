from torchvision import transforms, datasets
from PIL import Image
import glob
import random

input_size = 224
data_transforms = [
    transforms.RandomRotation(25),
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
]
print('made transform')

path = random.choice(list(glob.glob('./images/cv2/Three/Train/0/*')))
print('got path')
image = Image.open(path)
print('open file')
for transform in data_transforms:
    image = transform.forward(image)
print('transform')
image.save('./transformedImage.jpg')
print('sav')