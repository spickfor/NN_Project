#!/usr/bin/env python3

# this tests how well out model does on the test dataset

import os
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection_area
    return intersection_area / union_area

def point_to_box(x, y, radius=5):
    return [x - radius, y - radius, x + radius, y + radius]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'enhanced_cnn_model.pth'
    model = load_model(model_path).to(device)
    mtcnn = MTCNN(keep_all=True, device=device)

    folder_path = 'resized_gt_db'
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((384, 286)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    ious = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('L')
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                predicted_eyes = model(image_tensor).cpu().numpy().flatten()
                boxes, _, landmarks = mtcnn.detect(image.convert('RGB'), landmarks=True)
                if landmarks is not None and len(landmarks) > 0:
                    true_eyes = landmarks[0][:2]  # Assuming the first two landmarks are the eye positions
                    true_boxes = [point_to_box(x, y, 6) for x, y in true_eyes]
                    pred_boxes = [point_to_box(x, y, 6) for x, y in predicted_eyes.reshape(-1, 2)]
                    iou_scores = [calculate_iou(true_box, pred_box) for true_box, pred_box in zip(true_boxes, pred_boxes)]
                    avg_iou = np.mean(iou_scores)
                    ious.append(avg_iou)
                    print(f"IoU for {filename}: {avg_iou}")

    if ious:
        print("Average IoU across all images:", np.mean(ious))
    else:
        print("No IoUs calculated, possibly due to detection failures.")

if __name__ == "__main__":
    main()
