#!/usr/bin/env python3


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
from PIL import Image
from tqdm import tqdm


import matplotlib.pyplot as plt
import numpy as np
# from sklearn.model_selection import ParameterGrid




# this is our data loading and preprocessing
class BioIDFaceDataset(Dataset):
   # data_folder is the folder that holds the BioID data, transform is for data transformation
   def __init__(self, data_folder, transform=None):
       self.root_dir = data_folder
       self.transform = transform
       self.image_files = sorted(glob.glob(os.path.join(os.getcwd(),data_folder, '*.pgm')))
       self.eye_files = sorted(glob.glob(os.path.join(os.getcwd(),data_folder, '*.eye')))


   def __len__(self):
       return len(self.image_files)


   def __getitem__(self, idx):
       image_path = self.image_files[idx]
       eye_path = self.eye_files[idx]
      
       # Load image
       image = Image.open(image_path).convert('L')  # Convert to grayscale
      
       lx = ly = rx = ry = 0


       # Load eye positions, ignoring lines starting with '#'
       with open(eye_path, 'r') as f:
           for line in f:
               if not line.startswith('#'):
                   lx, ly, rx, ry = line.split()


       # Convert eye positions to tensor
       eye_positions = torch.tensor([float(lx), float(ly), float(rx), float(ry)])
      
       if self.transform:
           image = self.transform(image)
      
       return image, eye_positions



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


# Function to convert points to bounding boxes
def point_to_box(x, y, radius=5):
    return [x - radius, y - radius, x + radius, y + radius]

# Function to calculate IoU
def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area


# Define transformations (normalization and resizing)
transform = transforms.Compose([
   transforms.Resize((384, 286)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.5], std=[0.5])
])

# Create the dataset
dataset = BioIDFaceDataset(data_folder='BioID-FaceDatabase-V1', transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# testing to make sure loaded properly
print(f'Length of dataset: {len(dataset)}')


  
# Model create
model = SimpleCNN()
# loss func
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
radius = 5      # our radius for IoU 
num_epochs = 15
for epoch in range(num_epochs):
    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0
    for images, labels in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            predicted_eye_positions = output.view(-1, 2, 2)
            ground_truth_eye_positions = labels.view(-1, 2, 2)
            for i in range(predicted_eye_positions.size(0)):
                iou_sum = 0.0
                for j in range(2):
                    pred_box = point_to_box(predicted_eye_positions[i, j, 0], predicted_eye_positions[i, j, 1], radius)
                    true_box = point_to_box(ground_truth_eye_positions[i, j, 0], ground_truth_eye_positions[i, j, 1], radius)
                    iou_sum += calculate_iou(pred_box, true_box)
                total_iou += iou_sum / 2
    epoch_loss = total_loss / total_samples
    epoch_iou = total_iou / total_samples
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, IoU: {epoch_iou}')



# Save the model's state dictionary
torch.save(model.state_dict(), 'simple_cnn_model.pth')


