#!/usr/bin/env python3

# this uses a deeper CNN than the main function for the model

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Dataset class for image loading and preprocessing
class BioIDFaceDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.root_dir = data_folder
        self.transform = transform
        self.image_files = sorted(glob.glob(os.path.join(os.getcwd(), data_folder, '*.pgm')))
        self.eye_files = sorted(glob.glob(os.path.join(os.getcwd(), data_folder, '*.eye')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        eye_path = self.eye_files[idx]
        image = Image.open(image_path).convert('L')
        with open(eye_path, 'r') as f:
            lx, ly, rx, ry = map(float, next(filter(lambda x: not x.startswith('#'), f)).split())
        eye_positions = torch.tensor([lx, ly, rx, ry])
        if self.transform:
            image = self.transform(image)
        return image, eye_positions

# Enhanced CNN model definition
class EnhancedSimpleCNN(nn.Module):
    def __init__(self):
        super(EnhancedSimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 24 * 17, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.bn4(self.conv4(x)), 2))
        x = x.view(-1, 256 * 24 * 17)
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(x)

# Main function to execute the training and validation
def main():
    # transform = transforms.Compose([
    #     transforms.Resize((384, 286)),
    #     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
    #     transforms.RandomAdjustSharpness(sharpness_factor=2),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    # ])

    transform = transforms.Compose([
        transforms.Resize((384, 286)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = BioIDFaceDataset(data_folder='BioID-FaceDatabase-V1', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    model = EnhancedSimpleCNN().to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_model(model, train_loader, val_loader, criterion, optimizer)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0.0
        total_val_loss = 0.0
        total_train_samples = 0
        total_val_samples = 0

        # Training phase
        for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)
                total_val_samples += images.size(0)

        avg_train_loss = total_train_loss / total_train_samples
        avg_val_loss = total_val_loss / total_val_samples
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    torch.save(model.state_dict(), 'enhanced_cnn_model.pth')
    print("Model trained and saved.")

if __name__ == "__main__":
    main()
