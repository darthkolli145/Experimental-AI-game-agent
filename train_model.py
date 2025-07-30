import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mole_detection_model import MoleDetectionCNN
import json
import time

class MoleDataset(Dataset):
    """
    Custom dataset for mole vs hole classification
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

class DataCollector:
    """
    Collect training data from the game
    """
    def __init__(self, save_dir="training_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "holes"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "moles"), exist_ok=True)
        
        self.hole_count = 0
        self.mole_count = 0
        
    def save_sample(self, image_crop, is_mole=False):
        """
        Save a training sample
        """
        if is_mole:
            filename = f"mole_{self.mole_count:04d}.png"
            filepath = os.path.join(self.save_dir, "moles", filename)
            self.mole_count += 1
        else:
            filename = f"hole_{self.hole_count:04d}.png"
            filepath = os.path.join(self.save_dir, "holes", filename)
            self.hole_count += 1
        
        cv2.imwrite(filepath, image_crop)
        return filepath

def create_training_data():
    """
    Create training data by extracting grid cells from game screenshots
    """
    print("=== Training Data Creation ===")
    print("This will help you collect training data for the neural network.")
    print("You'll need to manually label whether each cell contains a mole or hole.")
    
    from ai_game_player import WhackAMoleAI
    
    ai = WhackAMoleAI(debug_mode=False)
    if not ai.setup():
        print("Setup failed. Cannot collect training data.")
        return
    
    collector = DataCollector()
    
    print("\nStarting data collection...")
    print("Instructions:")
    print("- Look at each grid cell shown")
    print("- Press 'm' if it contains a mole")
    print("- Press 'h' if it's just a hole")
    print("- Press 'q' to quit")
    
    samples_collected = 0
    while samples_collected < 200:  # Collect 200 samples
        # Capture game screen
        screenshot = ai.capture_game_screen()
        if screenshot is None:
            break
        
        # Get grid positions
        grid_positions = ai.detection_model.simple_detector.detect_grid_positions(screenshot)
        
        for i, (x, y) in enumerate(grid_positions):
            # Extract grid cell
            cell_size = 60
            half_size = cell_size // 2
            
            if (y - half_size >= 0 and y + half_size < screenshot.shape[0] and
                x - half_size >= 0 and x + half_size < screenshot.shape[1]):
                
                cell = screenshot[y-half_size:y+half_size, x-half_size:x+half_size]
                
                # Resize for consistency
                cell = cv2.resize(cell, (64, 64))
                
                # Show the cell to user
                cv2.imshow(f'Grid Cell {i} - Press m(mole) h(hole) q(quit)', cell)
                
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('m'):
                    # Save as mole
                    collector.save_sample(cell, is_mole=True)
                    samples_collected += 1
                    print(f"Saved mole sample {samples_collected}")
                elif key == ord('h'):
                    # Save as hole
                    collector.save_sample(cell, is_mole=False)
                    samples_collected += 1
                    print(f"Saved hole sample {samples_collected}")
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                
                cv2.destroyAllWindows()
        
        time.sleep(1)  # Wait before next screenshot
    
    print(f"\nData collection complete!")
    print(f"Moles: {collector.mole_count}")
    print(f"Holes: {collector.hole_count}")

def load_training_data(data_dir="training_data"):
    """
    Load training data from the collected samples
    """
    image_paths = []
    labels = []
    
    # Load hole samples (label 0)
    hole_dir = os.path.join(data_dir, "holes")
    if os.path.exists(hole_dir):
        for filename in os.listdir(hole_dir):
            if filename.endswith('.png'):
                image_paths.append(os.path.join(hole_dir, filename))
                labels.append(0)
    
    # Load mole samples (label 1)
    mole_dir = os.path.join(data_dir, "moles")
    if os.path.exists(mole_dir):
        for filename in os.listdir(mole_dir):
            if filename.endswith('.png'):
                image_paths.append(os.path.join(mole_dir, filename))
                labels.append(1)
    
    return image_paths, labels

def train_model(data_dir="training_data", epochs=50, batch_size=16):
    """
    Train the mole detection CNN model
    """
    print("=== Training Mole Detection Model ===")
    
    # Load data
    image_paths, labels = load_training_data(data_dir)
    
    if len(image_paths) == 0:
        print("No training data found! Please collect data first.")
        return
    
    print(f"Found {len(image_paths)} training samples")
    print(f"Holes: {labels.count(0)}, Moles: {labels.count(1)}")
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MoleDataset(train_paths, train_labels, train_transform)
    val_dataset = MoleDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoleDetectionCNN(num_classes=2)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"Training on {device}")
    print("Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        scheduler.step()
    
    # Save the trained model
    model_path = 'mole_detection_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model

def main():
    """
    Main training interface
    """
    print("=== Mole Detection Model Training ===")
    print("1. Collect training data")
    print("2. Train model")
    print("3. Both (collect then train)")
    
    choice = input("Choose option (1, 2, or 3): ").strip()
    
    if choice == "1":
        create_training_data()
    elif choice == "2":
        train_model()
    elif choice == "3":
        create_training_data()
        input("Press Enter to start training...")
        train_model()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 