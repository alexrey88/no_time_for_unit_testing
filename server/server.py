import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image

DATA_PATH = "./dataset"
MODEL_PATH = './model.pth'
NUM_LABELS = 4
SIDE_SIZE = 255
CROP_SIZE = 224

# Process our image
def process_image(image_path):
    # Load Image
    img = Image.open(image_path)
    
    # Get the dimensions of the image
    width, height = img.size
    
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is SIDE_SIZEpx
    img = img.resize((SIDE_SIZE, int(SIDE_SIZE*(height/width))) if width < height else (int(SIDE_SIZE*(width/height)), SIDE_SIZE))
    
    # Get the dimensions of the new image size
    width, height = img.size
    
    # Set the coordinates to do a center crop of CROP_SIZE x CROP_SIZE
    left = (width - CROP_SIZE)/2
    top = (height - CROP_SIZE)/2
    right = (width + CROP_SIZE)/2
    bottom = (height + CROP_SIZE)/2
    img = img.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    img = np.array(img)
    
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/SIDE_SIZE
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]

    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image

# Using our model to predict the label
def model_predict(image, model):
    # Pass the image through our model
    output = model.forward(image)   
    
    # Reverse the log function in our output
    output = torch.exp(output) 

    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

def main():
    classes = ['bench', 'other', 'picnic', 'trash']
    model = models.densenet161(pretrained=True)

    # Turn off training for their parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create new classifier for model using torch.nn as nn library
    classifier_input = model.classifier.in_features
    classifier = nn.Sequential( nn.Linear(classifier_input, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Linear(512, NUM_LABELS),
                                nn.LogSoftmax(dim=1))

    # Replace default classifier with new classifier
    model.classifier = classifier
    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    # Process Image
    import glob
    import os

    list_of_files = glob.glob('./nodeServer/uploads/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    image = process_image("./nodeServer/uploads/"+latest_file)

    # Give image to model to predict output
    top_prob, top_class = model_predict(image, model)

    print(top_class)

    if top_prob > 0.95:
        exit(classes[top_class])
    else:
        exit('other')

if __name__== "__main__":
    main()