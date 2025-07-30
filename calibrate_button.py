#!/usr/bin/env python3
"""
Button Detection Calibration Tool

This tool helps you calibrate the color detection for the START GAME button
in the whack-a-mole game.
"""

import cv2
import numpy as np
from ai_game_player import WhackAMoleAI
import time

def calibrate_start_button():
    """
    Interactive tool to calibrate START GAME button detection
    """
    print("=== START GAME Button Calibration ===")
    print("This tool will help you find the right color settings for the button")
    
    # Initialize AI to get game window
    ai = WhackAMoleAI(debug_mode=True)
    if not ai.setup():
        print("Could not setup game window")
        return
    
    print("\nMake sure the game is showing the START GAME button")
    input("Press Enter when the START GAME button is visible...")
    
    # Capture current screen
    screenshot = ai.capture_game_screen()
    if screenshot is None:
        print("Could not capture game screen")
        return
    
    # Save screenshot for reference
    cv2.imwrite('button_calibration.png', screenshot)
    print("Saved screenshot as 'button_calibration.png'")
    
    # Create trackbar window
    cv2.namedWindow('Button Detection Calibration')
    cv2.namedWindow('Original')
    cv2.namedWindow('Green Mask')
    
    # Initial values for green button detection
    cv2.createTrackbar('H Min', 'Button Detection Calibration', 35, 179, lambda x: None)
    cv2.createTrackbar('S Min', 'Button Detection Calibration', 50, 255, lambda x: None)
    cv2.createTrackbar('V Min', 'Button Detection Calibration', 50, 255, lambda x: None)
    cv2.createTrackbar('H Max', 'Button Detection Calibration', 85, 179, lambda x: None)
    cv2.createTrackbar('S Max', 'Button Detection Calibration', 255, 255, lambda x: None)
    cv2.createTrackbar('V Max', 'Button Detection Calibration', 255, 255, lambda x: None)
    
    cv2.createTrackbar('Min Area', 'Button Detection Calibration', 500, 20000, lambda x: None)
    cv2.createTrackbar('Max Area', 'Button Detection Calibration', 15000, 20000, lambda x: None)
    
    print("\nAdjust the trackbars to highlight the START GAME button")
    print("The button should appear white in the 'Green Mask' window")
    print("Green rectangles in the original image show detected buttons")
    print("Press 's' to save settings, 'q' to quit")
    
    while True:
        # Get trackbar values
        h_min = cv2.getTrackbarPos('H Min', 'Button Detection Calibration')
        s_min = cv2.getTrackbarPos('S Min', 'Button Detection Calibration')
        v_min = cv2.getTrackbarPos('V Min', 'Button Detection Calibration')
        h_max = cv2.getTrackbarPos('H Max', 'Button Detection Calibration')
        s_max = cv2.getTrackbarPos('S Max', 'Button Detection Calibration')
        v_max = cv2.getTrackbarPos('V Max', 'Button Detection Calibration')
        
        min_area = cv2.getTrackbarPos('Min Area', 'Button Detection Calibration')
        max_area = cv2.getTrackbarPos('Max Area', 'Button Detection Calibration')
        
        # Convert to HSV
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        
        # Create mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours and draw them
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw detection results
        result_img = screenshot.copy()
        button_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check if it looks like a button (reasonable aspect ratio)
                if 1.5 < aspect_ratio < 5:
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(result_img, f"BTN {button_count}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    button_count += 1
        
        # Add info text
        info_text = f"HSV: [{h_min},{s_min},{v_min}] to [{h_max},{s_max},{v_max}] | Area: {min_area}-{max_area} | Buttons: {button_count}"
        cv2.putText(result_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result_img, "Press 's' to save, 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show images
        cv2.imshow('Original', result_img)
        cv2.imshow('Green Mask', mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the calibrated values
            settings = {
                'h_min': h_min, 's_min': s_min, 'v_min': v_min,
                'h_max': h_max, 's_max': s_max, 'v_max': v_max,
                'min_area': min_area, 'max_area': max_area
            }
            
            print(f"\n=== CALIBRATED BUTTON DETECTION SETTINGS ===")
            print(f"HSV Lower: [{h_min}, {s_min}, {v_min}]")
            print(f"HSV Upper: [{h_max}, {s_max}, {v_max}]")
            print(f"Area Range: {min_area} - {max_area}")
            print(f"Detected {button_count} button(s)")
            
            # Save to file
            with open('button_settings.txt', 'w') as f:
                f.write(f"# START GAME Button Detection Settings\n")
                f.write(f"green_lower = np.array([{h_min}, {s_min}, {v_min}])\n")
                f.write(f"green_upper = np.array([{h_max}, {s_max}, {v_max}])\n")
                f.write(f"min_area = {min_area}\n")
                f.write(f"max_area = {max_area}\n")
            
            print("Settings saved to 'button_settings.txt'")
            print("You can copy these values into the ai_game_player.py file")
            break
    
    cv2.destroyAllWindows()

def test_current_detection():
    """
    Test the current button detection without modification
    """
    print("=== Testing Current Button Detection ===")
    
    ai = WhackAMoleAI(debug_mode=True)
    if not ai.setup():
        print("Could not setup game window")
        return
    
    print("Make sure the START GAME button is visible")
    input("Press Enter to test detection...")
    
    screenshot = ai.capture_game_screen()
    if screenshot is None:
        print("Could not capture screen")
        return
    
    # Test the green detection method
    print("Testing green color detection...")
    success = ai._detect_start_button_green(screenshot)
    
    if success:
        print("✅ Button detection successful!")
    else:
        print("❌ Button detection failed")
        print("Try the calibration tool to adjust settings")

def main():
    print("🔨 START GAME Button Detection Tool 🤖")
    print("=" * 50)
    print("1. Test current detection")
    print("2. Calibrate button detection")
    print("3. Exit")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        test_current_detection()
    elif choice == "2":
        calibrate_start_button()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 