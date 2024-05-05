#!/usr/bin/env python3


# this uses mtcnn to put eye points on a face and then puts our prediction from our model on their as well, but this uses the 
# deeper CNN model
import cv2
from mtcnn import MTCNN
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Model class (must be identical to the one used in training)
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

# Function to load the model
def load_model(model_path):
    model = EnhancedSimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Initialize MTCNN detector
detector = MTCNN()

# Load an image
# image_path = 'BioID_0000.pgm'         # from validation set
# image_path = 'BioID_0016 copy.pgm'    # from training set
# image_path = 'resized_test.jpeg'
image_path = 's08_01.jpg'               # from test dataset

image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces in the image using MTCNN
results = detector.detect_faces(rgb_image)

# Process each face found
for result in results:
    bounding_box = result['box']
    keypoints = result['keypoints']
    cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), (0,155,255), 2)
    for key, point in keypoints.items():
        cv2.circle(image, point, 2, (255,0,0), 2)

# Load and prepare the image for the SimpleCNN model
pil_image = Image.open(image_path).convert('L')

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((384, 286)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
image_tensor = transform(pil_image).unsqueeze(0)

# Load the model and make a prediction
model = load_model('enhanced_cnn_model.pth')

with torch.no_grad():
    prediction = model(image_tensor)
    predicted_positions = prediction[0].numpy()  # Convert to numpy array
    predicted_positions = predicted_positions.reshape(2, 2)  # Reshape to (2,2) for eye coordinates

# Draw predicted eye positions from the SimpleCNN model
for (x, y) in predicted_positions:
    x, y = int(x), int(y)
    cv2.circle(image, (x, y), 2, (0, 0, 255), 2)

# Show the combined keypoints
cv2.imshow('Image with MTCNN and SimpleCNN predictions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
