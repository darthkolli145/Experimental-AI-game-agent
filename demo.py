#!/usr/bin/env python3
"""
Demo script for the Whack-A-Mole Computer Vision System

This script demonstrates the mole detection capabilities without requiring
the full game setup. Great for testing and development.
"""

import cv2
import numpy as np
import argparse
import os
from mole_detection_model import MoleDetectionModel, SimpleMoleDetector
import time

def demo_webcam():
    """
    Demo mole detection using webcam feed
    """
    print("=== Webcam Demo ===")
    print("This will test mole detection on your webcam feed")
    print("Hold up objects or images that look like moles/holes")
    print("Press 'q' to quit")
    
    detector = SimpleMoleDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect moles in the frame
        mole_positions = detector.detect_moles_simple(frame)
        
        # Visualize detections
        for x, y in mole_positions:
            cv2.circle(frame, (x, y), 30, (0, 255, 0), 3)
            cv2.putText(frame, "MOLE DETECTED", (x-50, y-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add info text
        cv2.putText(frame, f"Moles detected: {len(mole_positions)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Mole Detection Demo - Webcam', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def demo_image(image_path):
    """
    Demo mole detection on a static image
    """
    print(f"=== Image Demo: {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    
    detector = SimpleMoleDetector()
    
    # Detect moles
    mole_positions = detector.detect_moles_simple(image)
    grid_positions = detector.detect_grid_positions(image)
    
    # Create visualization
    vis_image = image.copy()
    
    # Draw detected moles
    for x, y in mole_positions:
        cv2.circle(vis_image, (x, y), 20, (0, 255, 0), 3)
        cv2.putText(vis_image, "MOLE", (x-20, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw grid positions
    for i, (x, y) in enumerate(grid_positions):
        cv2.circle(vis_image, (x, y), 5, (255, 0, 0), 2)
        cv2.putText(vis_image, str(i), (x+10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Add detection count
    cv2.putText(vis_image, f"Moles detected: {len(mole_positions)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show results
    cv2.imshow('Mole Detection Demo - Image', vis_image)
    print(f"Detected {len(mole_positions)} moles")
    print("Press any key to close the window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_color_calibration():
    """
    Interactive color calibration tool
    """
    print("=== Color Calibration Demo ===")
    print("This tool helps you calibrate color ranges for mole detection")
    print("Use trackbars to adjust HSV color ranges")
    print("Press 'q' to quit, 's' to save current values")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create window and trackbars
    cv2.namedWindow('Color Calibration')
    cv2.namedWindow('Mask')
    
    # HSV ranges (Hue, Saturation, Value)
    cv2.createTrackbar('H Min', 'Color Calibration', 15, 179, lambda x: None)
    cv2.createTrackbar('S Min', 'Color Calibration', 100, 255, lambda x: None)
    cv2.createTrackbar('V Min', 'Color Calibration', 100, 255, lambda x: None)
    cv2.createTrackbar('H Max', 'Color Calibration', 35, 179, lambda x: None)
    cv2.createTrackbar('S Max', 'Color Calibration', 255, 255, lambda x: None)
    cv2.createTrackbar('V Max', 'Color Calibration', 255, 255, lambda x: None)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get trackbar values
        h_min = cv2.getTrackbarPos('H Min', 'Color Calibration')
        s_min = cv2.getTrackbarPos('S Min', 'Color Calibration')
        v_min = cv2.getTrackbarPos('V Min', 'Color Calibration')
        h_max = cv2.getTrackbarPos('H Max', 'Color Calibration')
        s_max = cv2.getTrackbarPos('S Max', 'Color Calibration')
        v_max = cv2.getTrackbarPos('V Max', 'Color Calibration')
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply mask
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Add text with current values
        text = f"HSV Lower: [{h_min}, {s_min}, {v_min}] Upper: [{h_max}, {s_max}, {v_max}]"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Color Calibration', frame)
        cv2.imshow('Mask', mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\nSaved color values:")
            print(f"Lower HSV: [{h_min}, {s_min}, {v_min}]")
            print(f"Upper HSV: [{h_max}, {s_max}, {v_max}]")
            print("You can use these values in the SimpleMoleDetector class")
    
    cap.release()
    cv2.destroyAllWindows()

def create_test_image():
    """
    Create a synthetic test image that looks like the game
    """
    print("=== Creating Test Image ===")
    
    # Create a green background (like the game)
    img = np.zeros((600, 500, 3), dtype=np.uint8)
    img[:] = (87, 139, 46)  # Forest green background
    
    # Define 3x3 grid positions
    grid_size = 60
    start_x, start_y = 100, 150
    spacing = 120
    
    # Draw holes and some moles
    for row in range(3):
        for col in range(3):
            x = start_x + col * spacing
            y = start_y + row * spacing
            
            # Draw hole (brown circle)
            cv2.circle(img, (x, y), grid_size//2, (19, 69, 139), -1)  # Brown hole
            
            # Add some moles (golden circles) randomly
            if (row, col) in [(0, 1), (1, 2), (2, 0)]:  # Some positions have moles
                cv2.circle(img, (x, y), grid_size//3, (0, 215, 255), -1)  # Golden mole
    
    # Add title
    cv2.putText(img, "Test Image for Mole Detection", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save image
    cv2.imwrite('test_game_image.png', img)
    print("Created test_game_image.png")
    
    return 'test_game_image.png'

def main():
    """
    Main demo function
    """
    parser = argparse.ArgumentParser(description='Whack-A-Mole Computer Vision Demo')
    parser.add_argument('--mode', choices=['webcam', 'image', 'calibrate', 'test'], 
                       default='test', help='Demo mode to run')
    parser.add_argument('--image', type=str, help='Path to image file for image mode')
    
    args = parser.parse_args()
    
    print("🔨 Whack-A-Mole Computer Vision Demo 🤖")
    print("=" * 50)
    
    if args.mode == 'webcam':
        demo_webcam()
    elif args.mode == 'image':
        if args.image:
            demo_image(args.image)
        else:
            print("Please provide --image path for image mode")
    elif args.mode == 'calibrate':
        demo_color_calibration()
    elif args.mode == 'test':
        print("Creating test image and running detection...")
        test_image = create_test_image()
        time.sleep(1)
        demo_image(test_image)
    
    print("\nDemo complete! 🎯")

if __name__ == "__main__":
    main() 