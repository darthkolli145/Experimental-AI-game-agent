import cv2
import numpy as np
import pyautogui
import time
import threading
from mss import mss
from mole_detection_model import MoleDetectionModel
import tkinter as tk
from tkinter import messagebox
import os

class WhackAMoleAI:
    """
    AI agent that automatically plays the whack-a-mole game
    """
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.running = False
        self.game_window_region = None
        self.detection_model = MoleDetectionModel()
        
        # Screen capture setup
        self.sct = mss()
        
        # Click delay to avoid too rapid clicking
        self.click_delay = 0.1
        self.last_click_time = 0
        
        # Performance tracking
        self.moles_detected = 0
        self.moles_clicked = 0
        self.start_time = None
        
        # Configure pyautogui
        pyautogui.PAUSE = 0.05  # Small pause between actions
        pyautogui.FAILSAFE = True  # Move mouse to corner to stop
        
        print("WhackAMole AI initialized!")
        print("Move mouse to top-left corner to emergency stop")
        
    def find_game_window(self):
        """
        Find the whack-a-mole game window on screen
        """
        # Take a screenshot to find the game window
        screenshot = np.array(self.sct.grab(self.sct.monitors[1]))
        
        # Convert BGRA to BGR
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        # Look for the game window by detecting the characteristic green background
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        
        # Green color range for the game background
        green_lower = np.array([60, 100, 50])
        green_upper = np.array([80, 255, 255])
        
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest green region (likely the game window)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Validate size (game window should be reasonable size)
            if w > 300 and h > 400:
                self.game_window_region = {"top": y, "left": x, "width": w, "height": h}
                print(f"Found game window at: {self.game_window_region}")
                return True
        
        print("Could not find game window automatically.")
        print("Please make sure the whack-a-mole game is visible on screen.")
        return False
    
    def manual_window_selection(self):
        """
        Let user manually select the game window region
        """
        print("\n=== Manual Window Selection ===")
        print("1. Make sure the whack-a-mole game is visible")
        print("2. Look at the game window coordinates")
        print("3. Enter the approximate region")
        
        try:
            x = int(input("Enter game window X coordinate (left edge): "))
            y = int(input("Enter game window Y coordinate (top edge): "))
            w = int(input("Enter game window width: "))
            h = int(input("Enter game window height: "))
            
            self.game_window_region = {"top": y, "left": x, "width": w, "height": h}
            print(f"Set game window region: {self.game_window_region}")
            return True
        except ValueError:
            print("Invalid input. Please enter numbers only.")
            return False
    
    def capture_game_screen(self):
        """
        Capture the current game window
        """
        if not self.game_window_region:
            return None
            
        # Capture the game window region
        screenshot = np.array(self.sct.grab(self.game_window_region))
        
        # Convert BGRA to BGR for OpenCV
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        return screenshot
    
    def click_mole(self, x, y):
        """
        Click on a mole at the given coordinates
        """
        current_time = time.time()
        
        # Avoid clicking too rapidly
        if current_time - self.last_click_time < self.click_delay:
            return
        
        # Convert relative coordinates to absolute screen coordinates
        abs_x = self.game_window_region["left"] + x
        abs_y = self.game_window_region["top"] + y
        
        try:
            # Quick click
            pyautogui.click(abs_x, abs_y)
            self.moles_clicked += 1
            self.last_click_time = current_time
            
            if self.debug_mode:
                print(f"Clicked mole at ({abs_x}, {abs_y})")
                
        except pyautogui.FailSafeException:
            print("Emergency stop activated!")
            self.running = False
    
    def start_game_automatically(self):
        """
        Try to automatically click the START GAME button
        """
        screenshot = self.capture_game_screen()
        if screenshot is None:
            return False
        
        # Try multiple methods to find the start button
        methods = [
            self._detect_start_button_green,
            self._detect_start_button_text,
            self._detect_start_button_manual_assist
        ]
        
        for method in methods:
            if method(screenshot):
                return True
        
        print("Could not find START GAME button with any method")
        print("Please manually click START GAME in the game window")
        input("Press Enter after you've started the game...")
        return True
    
    def _detect_start_button_green(self, screenshot):
        """Try to detect start button by green color"""
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        
        # Try multiple green color ranges
        green_ranges = [
            ([50, 100, 100], [70, 255, 255]),   # Original range
            ([40, 80, 80], [80, 255, 255]),     # Wider range
            ([35, 50, 50], [85, 255, 255]),     # Even wider
        ]
        
        for green_lower, green_upper in green_ranges:
            green_mask = cv2.inRange(hsv, np.array(green_lower), np.array(green_upper))
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 15000:  # Wider button size range
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (buttons are usually wider than tall)
                    aspect_ratio = w / h
                    if 1.5 < aspect_ratio < 5:  # Reasonable button proportions
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Click the start button
                        abs_x = self.game_window_region["left"] + center_x
                        abs_y = self.game_window_region["top"] + center_y
                        pyautogui.click(abs_x, abs_y)
                        print(f"Clicked START GAME button (green detection) at ({abs_x}, {abs_y})")
                        return True
        return False
    
    def _detect_start_button_text(self, screenshot):
        """Try to detect start button by looking for button-like rectangles"""
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Look for rectangular contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 20000:
                # Approximate the contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Look for rectangular shapes (4 corners)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Check if it's in the lower part of screen (where buttons usually are)
                    if (1.5 < aspect_ratio < 4 and 
                        y > screenshot.shape[0] * 0.6):  # Lower 40% of screen
                        
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        abs_x = self.game_window_region["left"] + center_x
                        abs_y = self.game_window_region["top"] + center_y
                        pyautogui.click(abs_x, abs_y)
                        print(f"Clicked START GAME button (shape detection) at ({abs_x}, {abs_y})")
                        return True
        return False
    
    def _detect_start_button_manual_assist(self, screenshot):
        """Show screenshot and let user point to start button"""
        if not self.debug_mode:
            return False
            
        print("Showing screenshot for manual button detection...")
        cv2.imshow('Find START GAME Button - Click on it', screenshot)
        print("Look at the screenshot, find the START GAME button")
        print("Estimate its center coordinates and close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        try:
            x = int(input("Enter X coordinate of START GAME button center: "))
            y = int(input("Enter Y coordinate of START GAME button center: "))
            
            abs_x = self.game_window_region["left"] + x
            abs_y = self.game_window_region["top"] + y
            pyautogui.click(abs_x, abs_y)
            print(f"Clicked START GAME button (manual) at ({abs_x}, {abs_y})")
            return True
        except ValueError:
            print("Invalid coordinates")
            return False
    
    def play_game(self, duration=30):
        """
        Main game playing loop
        """
        if not self.game_window_region:
            print("Game window not set. Please run setup first.")
            return
        
        print(f"\n=== Starting AI Game Player ===")
        print(f"Game duration: {duration} seconds")
        print("Move mouse to top-left corner for emergency stop")
        print("Starting in 3 seconds...")
        
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        # Try to start the game automatically
        self.start_game_automatically()
        time.sleep(1)  # Wait for game to start
        
        self.running = True
        self.start_time = time.time()
        self.moles_detected = 0
        self.moles_clicked = 0
        
        while self.running and (time.time() - self.start_time) < duration:
            try:
                # Capture current screen
                screenshot = self.capture_game_screen()
                if screenshot is None:
                    break
                
                # Detect moles
                mole_positions = self.detection_model.detect_moles(screenshot, use_cnn=False)
                
                if mole_positions:
                    self.moles_detected += len(mole_positions)
                    
                    # Click on all detected moles
                    for x, y in mole_positions:
                        if self.running:
                            self.click_mole(x, y)
                
                # Show debug visualization
                if self.debug_mode and mole_positions:
                    vis_image = self.detection_model.visualize_detections(screenshot, mole_positions)
                    cv2.imshow('AI Whack-A-Mole', vis_image)
                    cv2.waitKey(1)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.05)
                
            except pyautogui.FailSafeException:
                print("Emergency stop activated!")
                break
            except KeyboardInterrupt:
                print("Stopped by user")
                break
        
        self.running = False
        
        # Print performance statistics
        elapsed_time = time.time() - self.start_time
        print(f"\n=== Game Session Complete ===")
        print(f"Duration: {elapsed_time:.1f} seconds")
        print(f"Moles detected: {self.moles_detected}")
        print(f"Moles clicked: {self.moles_clicked}")
        if self.moles_detected > 0:
            accuracy = (self.moles_clicked / self.moles_detected) * 100
            print(f"Click accuracy: {accuracy:.1f}%")
        
        # Close debug window
        cv2.destroyAllWindows()
    
    def calibrate_detection(self):
        """
        Calibrate the mole detection by showing what the AI sees
        """
        if not self.game_window_region:
            print("Game window not set. Please run setup first.")
            return
        
        print("\n=== Detection Calibration Mode ===")
        print("This will show what the AI sees. Press 'q' to quit.")
        
        while True:
            screenshot = self.capture_game_screen()
            if screenshot is None:
                break
            
            # Detect moles and grid
            mole_positions = self.detection_model.detect_moles(screenshot, use_cnn=False)
            grid_positions = self.detection_model.simple_detector.detect_grid_positions(screenshot)
            
            # Visualize
            vis_image = self.detection_model.visualize_detections(
                screenshot, mole_positions, grid_positions
            )
            
            cv2.imshow('AI Vision - Press Q to quit', vis_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def setup(self):
        """
        Setup the AI player
        """
        print("=== WhackAMole AI Setup ===")
        print("1. Automatic window detection")
        print("2. Manual window selection")
        
        choice = input("Choose setup method (1 or 2): ").strip()
        
        if choice == "1":
            if not self.find_game_window():
                print("Automatic detection failed. Trying manual selection...")
                return self.manual_window_selection()
        elif choice == "2":
            return self.manual_window_selection()
        else:
            print("Invalid choice")
            return False
        
        return True

def main():
    """
    Main function to run the AI player
    """
    ai_player = WhackAMoleAI(debug_mode=True)
    
    print("=== WhackAMole AI Player ===")
    print("Make sure the whack-a-mole game is running and visible!")
    input("Press Enter when ready...")
    
    # Setup the AI
    if not ai_player.setup():
        print("Setup failed. Exiting.")
        return
    
    while True:
        print("\n=== Options ===")
        print("1. Play game (30 seconds)")
        print("2. Calibrate detection")
        print("3. Setup window region again")
        print("4. Exit")
        
        choice = input("Choose option: ").strip()
        
        if choice == "1":
            ai_player.play_game(30)
        elif choice == "2":
            ai_player.calibrate_detection()
        elif choice == "3":
            ai_player.setup()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 