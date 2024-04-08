#!/usr/bin/env python3
'''
run this script to:
Load the trained model and test on one validation sample.
'''


# Imports
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Model class (must be identical to the one used in train.py)
class SimpleCNN(nn.Module):
    def __init__(self):
       super(SimpleCNN, self).__init__()
       self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
       # Adjusting the size of the tensor to match the output dimensions needed
       self.fc1 = nn.Linear(64 * 96 * 71, 128)
       self.fc2 = nn.Linear(128, 4)  # Changed from 2 to 4 to output coordinates for both eyes




    def forward(self, x):
       x = F.relu(F.max_pool2d(self.conv1(x), 2))
       x = F.relu(F.max_pool2d(self.conv2(x), 2))
       x = x.view(-1, 64 * 96 * 71)  # Ensure this matches the output from the last pooling layer
       x = F.relu(self.fc1(x))
       x = self.fc2(x)  # Outputs 4 values
       return x


# Function to load the model
def load_model(model_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


# Main function to load the model and test it
def main():
    # Load the model
    model_path = 'simple_cnn_model.pth'
    model = load_model(model_path)
   
    # Load and preprocess a single image
    image_path = './BioID_0000.pgm'
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((384, 286)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image).unsqueeze(0)
   
    # Make a prediction
    with torch.no_grad():
        prediction = model(image_tensor)
   
    print("Predicted Eye Positions:", prediction.numpy())


if __name__ == "__main__":
    main()



