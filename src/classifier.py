import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Simple CNN architecture similar to what's used in Kaggle pneumonia competitions
class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: Normal, Pneumonia
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize model (in practice, you'd load pre-trained weights)
model = ImageCNN()
model.eval()

# Image preprocessing similar to Kaggle approaches
def preprocess_for_classification(image_path):
    """Preprocess image for classification."""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def classify_image(image_path):
    """
    Classify an image as Normal or Pneumonia.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: "Normal" or "Pneumonia"
    """
    try:
        # Preprocess image
        image_tensor = preprocess_for_classification(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Return classification
        return "Pneumonia" if predicted_class == 1 else "Normal"
        
    except Exception as e:
        print(f"Error classifying {image_path}: {e}")
        return "Normal"  # Default to Normal if classification fails
