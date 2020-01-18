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
NUM_LABELS = 3

def train_model():
    transformations = transforms.Compose([transforms.Resize(255),
        transforms.CenterCrop(224), 
        transforms.ToTensor(), # transform image in tensor object (separate R, G and B)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = ImageFolder(root=DATA_PATH, transform=transformations)
    print(data.classes)

    # Get train set and test set
    data_length = len(data)  # total number of examples
    test_length = int(0.4 * data_length)  # take ~ 30% for test
    test_set = torch.utils.data.Subset(data, range(test_length))  # take first 10%
    train_set = torch.utils.data.Subset(data, range(test_length, data_length))  # take the rest

    # Initialize train and test loaders
    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
    testloader = DataLoader(test_set, batch_size=32, shuffle=True) #drop_last=True,num_workers=2

    # Get pretrained model using torchvision.models as models library
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

    # Find the device available to use using torch library
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the device specified above
    model.to(device)

    # Set the error function using torch.nn as nn library
    criterion = nn.NLLLoss()

    # Set the optimizer function using torch.optim as optim library
    optimizer = optim.Adam(model.classifier.parameters())

    epochs = 10
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        accuracy = 0
        
        # Training the model
        model.train()
        counter = 0
        for inputs, labels in trainloader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear optimizers
            optimizer.zero_grad()
            # Forward pass
            output = model.forward(inputs)
            # Loss
            loss = criterion(output, labels)
            # Calculate gradients (backpropogation)
            loss.backward()
            # Adjust parameters based on gradients
            optimizer.step()
            # Add the loss to the training set's rnning loss
            train_loss += loss.item()*inputs.size(0)
            
            # Print the progress of our training
            counter += 1
            print(counter, "/", len(trainloader))
            
        # Evaluating the model
        model.eval()
        counter = 0
        # Tell torch not to calculate gradients
        with torch.no_grad():
            for inputs, labels in testloader:
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                output = model.forward(inputs)
                # Calculate Loss
                valloss = criterion(output, labels)
                # Add loss to the validation set's running loss
                val_loss += valloss.item()*inputs.size(0)
                
                # Since our model outputs a LogSoftmax, find the real 
                # percentages by reversing the log function
                output = torch.exp(output)
                # Get the top class of the output
                top_p, top_class = output.topk(1, dim=1)
                # See how many of the classes were correct?
                equals = top_class == labels.view(*top_class.shape)
                # Calculate the mean (get the accuracy for this batch)
                # and add it to the running accuracy for this epoch
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                # Print the progress of our evaluation
                counter += 1
                print(counter, "/", len(testloader))
        
        # Get the average loss for the entire epoch
        train_loss = train_loss/len(trainloader.dataset)
        valid_loss = val_loss/len(testloader.dataset)
        # Print out the information
        print('Accuracy: ', accuracy/len(testloader))
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))


def main():
    train_model()

if __name__== "__main__":
    main()