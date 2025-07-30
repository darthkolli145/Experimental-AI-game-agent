#!/usr/bin/env python3
"""
Launcher script for Whack-A-Mole: Human vs AI

This script provides an easy way to launch different game modes:
- Human only
- AI only  
- Human vs AI (side by side)
- Demo mode
"""

import subprocess
import threading
import time
import os
import sys

def run_human_game():
    """Launch the human whack-a-mole game"""
    print("🎮 Starting Human Game...")
    subprocess.run([sys.executable, "whack_a_mole.py"])

def run_ai_player():
    """Launch the AI player"""
    print("🤖 Starting AI Player...")
    subprocess.run([sys.executable, "ai_game_player.py"])

def run_demo():
    """Launch the computer vision demo"""
    print("🔍 Starting Computer Vision Demo...")
    subprocess.run([sys.executable, "demo.py"])

def run_training():
    """Launch the model training"""
    print("🧠 Starting Model Training...")
    subprocess.run([sys.executable, "train_model.py"])

def run_both():
    """Launch both human game and AI player"""
    print("👥 Starting Human vs AI Mode...")
    print("This will launch both the game and AI in separate windows")
    
    # Start human game in a separate thread
    human_thread = threading.Thread(target=run_human_game)
    human_thread.daemon = True
    human_thread.start()
    
    # Wait a bit for the game to load
    time.sleep(2)
    
    # Start AI player
    print("Game should be running. Now starting AI...")
    run_ai_player()

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import cv2
        import numpy as np
        import pyautogui
        import mss
        print("✅ All dependencies found!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def main():
    """Main launcher interface"""
    print("🔨 Whack-A-Mole: Human vs AI Launcher 🤖")
    print("=" * 50)
    
    # Check dependencies for AI modes
    deps_ok = check_dependencies()
    
    print("\nChoose your game mode:")
    print("1. 🎮 Human Game Only (no dependencies needed)")
    print("2. 🤖 AI Player Only (requires dependencies)")
    print("3. 👥 Human vs AI (side by side)")
    print("4. 🔍 Computer Vision Demo")
    print("5. 🧠 Train AI Model")
    print("6. ❓ Help & Requirements")
    print("7. 🚪 Exit")
    
    while True:
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            run_human_game()
            break
        elif choice == "2":
            if deps_ok:
                run_ai_player()
            else:
                print("Dependencies not found. Install them first or choose Human Game Only.")
            break
        elif choice == "3":
            if deps_ok:
                run_both()
            else:
                print("Dependencies not found. Install them first or choose Human Game Only.")
            break
        elif choice == "4":
            if deps_ok:
                run_demo()
            else:
                print("Dependencies not found. Install them first.")
            break
        elif choice == "5":
            if deps_ok:
                run_training()
            else:
                print("Dependencies not found. Install them first.")
            break
        elif choice == "6":
            show_help()
        elif choice == "7":
            print("Goodbye! 🐹💥")
            break
        else:
            print("Invalid choice. Please enter 1-7.")

def show_help():
    """Show help information"""
    print("\n" + "=" * 60)
    print("📖 HELP & REQUIREMENTS")
    print("=" * 60)
    
    print("\n🎮 HUMAN GAME ONLY:")
    print("- No additional dependencies needed")
    print("- Uses built-in Python tkinter")
    print("- Run: python whack_a_mole.py")
    
    print("\n🤖 AI PLAYER:")
    print("- Requires computer vision dependencies")
    print("- Install with: pip install -r requirements.txt")
    print("- Needs: PyTorch, OpenCV, NumPy, pyautogui, mss")
    
    print("\n👥 HUMAN vs AI MODE:")
    print("1. Launches the human game first")
    print("2. Position the game window where you want it")
    print("3. AI will try to detect the game window automatically")
    print("4. AI will play alongside you!")
    
    print("\n🔍 DEMO MODE:")
    print("- Test computer vision without the full game")
    print("- python demo.py --mode test (synthetic image)")
    print("- python demo.py --mode webcam (use your camera)")
    print("- python demo.py --mode calibrate (adjust colors)")
    
    print("\n🧠 TRAINING MODE:")
    print("- Collect training data for neural network")
    print("- Train custom PyTorch model")
    print("- Improve AI accuracy")
    
    print("\n💡 TIPS:")
    print("- For best AI performance, keep game window in consistent position")
    print("- Use good lighting and clear screen visibility")
    print("- Emergency stop: move mouse to top-left corner")
    print("- AI works best with color-based detection (default)")
    
    print("\n📁 FILES:")
    print("- whack_a_mole.py: Main game")
    print("- ai_game_player.py: AI player")
    print("- mole_detection_model.py: Computer vision models")
    print("- train_model.py: Neural network training")
    print("- demo.py: Computer vision demos")
    print("- requirements.txt: Dependencies list")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 