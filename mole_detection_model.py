import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np

class MoleDetectionCNN(nn.Module):
    """
    Convolutional Neural Network for detecting moles in whack-a-mole game
    """
    def __init__(self, num_classes=2):  # 0: hole, 1: mole
        super(MoleDetectionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # Input size after convolutions: 128 * 8 * 8 (for 64x64 input)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 16x16
        x = self.pool(F.relu(self.conv3(x)))  # 8x8
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class SimpleMoleDetector:
    """
    A simpler computer vision approach using color and template matching
    for quick prototyping before training the neural network
    """
    def __init__(self):
        self.hole_color_lower = np.array([10, 50, 20])   # Brown hole color range
        self.hole_color_upper = np.array([30, 255, 200])
        self.mole_color_lower = np.array([15, 100, 100])  # Golden mole color range
        self.mole_color_upper = np.array([35, 255, 255])
        
    def detect_moles_simple(self, image):
        """
        Simple mole detection using color thresholding
        Returns list of (x, y) coordinates where moles are detected
        """
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for mole colors (golden/yellow)
        mole_mask = cv2.inRange(hsv, self.mole_color_lower, self.mole_color_upper)
        
        # Find contours
        contours, _ = cv2.findContours(mole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mole_positions = []
        for contour in contours:
            # Filter by area to avoid noise
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    mole_positions.append((cx, cy))
        
        return mole_positions
    
    def detect_grid_positions(self, image):
        """
        Detect the 3x3 grid positions in the game
        Returns list of (x, y) coordinates for each grid cell center
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use template matching or contour detection to find grid
        # For now, we'll estimate based on image dimensions
        height, width = image.shape[:2]
        
        grid_positions = []
        for row in range(3):
            for col in range(3):
                # Estimate grid positions (adjust based on your game window)
                x = width // 6 + col * (width // 3)
                y = height // 4 + row * (height // 4)
                grid_positions.append((x, y))
        
        return grid_positions

class MoleDetectionModel:
    """
    Main model class that combines CNN and simple detection methods
    """
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MoleDetectionCNN()
        self.model.to(self.device)
        
        self.simple_detector = SimpleMoleDetector()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if model_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                print(f"Loaded model from {model_path}")
            except:
                print(f"Could not load model from {model_path}, using untrained model")
    
    def preprocess_grid_cell(self, image, x, y, cell_size=60):
        """
        Extract and preprocess a grid cell for the CNN
        """
        half_size = cell_size // 2
        cell = image[y-half_size:y+half_size, x-half_size:x+half_size]
        
        if cell.shape[0] == 0 or cell.shape[1] == 0:
            return None
            
        # Resize if necessary
        cell = cv2.resize(cell, (64, 64))
        
        # Convert BGR to RGB
        cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(cell_rgb)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict_mole(self, image, x, y):
        """
        Predict if there's a mole at position (x, y) using the CNN
        """
        cell_tensor = self.preprocess_grid_cell(image, x, y)
        if cell_tensor is None:
            return False
            
        with torch.no_grad():
            outputs = self.model(cell_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return predicted_class == 1 and confidence > 0.7  # Class 1 is mole
    
    def detect_moles(self, image, use_cnn=False):
        """
        Main detection method - can use either simple detection or CNN
        """
        if use_cnn:
            # Use CNN approach
            grid_positions = self.simple_detector.detect_grid_positions(image)
            mole_positions = []
            
            for x, y in grid_positions:
                if self.predict_mole(image, x, y):
                    mole_positions.append((x, y))
                    
            return mole_positions
        else:
            # Use simple color-based detection
            return self.simple_detector.detect_moles_simple(image)
    
    def visualize_detections(self, image, mole_positions, grid_positions=None):
        """
        Draw detection results on the image for debugging
        """
        vis_image = image.copy()
        
        # Draw detected moles
        for x, y in mole_positions:
            cv2.circle(vis_image, (x, y), 20, (0, 255, 0), 3)
            cv2.putText(vis_image, "MOLE", (x-20, y-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw grid positions if provided
        if grid_positions:
            for i, (x, y) in enumerate(grid_positions):
                cv2.circle(vis_image, (x, y), 5, (255, 0, 0), 2)
                cv2.putText(vis_image, str(i), (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return vis_image 