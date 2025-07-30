# 🔨 Whack-A-Mole Game + AI Player

A fun and interactive Whack-A-Mole game built with Python and tkinter, plus an AI agent that can automatically play the game using computer vision and PyTorch!

## 🎮 Game Features

### Human Player Mode
- **3x3 Grid**: 9 holes where moles can appear
- **Random Spawning**: Moles appear at random locations and times
- **Timer**: 30-second countdown
- **Score Tracking**: Earn 10 points per mole whacked
- **Visual Effects**: Explosion animation when you successfully whack a mole
- **Beautiful UI**: Colorful interface with emojis

### AI Player Mode
- **Computer Vision**: AI detects moles using OpenCV and PyTorch
- **Automatic Clicking**: AI automatically clicks on detected moles
- **Performance Tracking**: Monitor AI's detection accuracy and click rate
- **Debug Visualization**: See what the AI sees in real-time
- **Multiple Detection Methods**: Color-based detection and neural network options

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Human Game
```bash
python whack_a_mole.py
```

### 3. Run the AI Player
```bash
python ai_game_player.py
```

## 🎯 How to Play (Human Mode)

1. Click the **START GAME** button to begin
2. Moles (🐹) will randomly appear in the holes
3. Click on the moles quickly to whack them before they disappear
4. Each successful whack earns you 10 points
5. You have 30 seconds to get the highest score possible
6. When you successfully whack a mole, you'll see a 💥 explosion effect

## 🤖 How to Use the AI Player

### Setup
1. Run the whack-a-mole game: `python whack_a_mole.py`
2. In a new terminal, run the AI: `python ai_game_player.py`
3. The AI will help you locate the game window automatically
4. Choose from the menu options:
   - **Play game**: AI plays automatically for 30 seconds
   - **Calibrate detection**: See what the AI detects
   - **Setup window**: Reconfigure game window location

### AI Options
- **Automatic Window Detection**: AI finds the game window
- **Manual Window Selection**: Manually specify window coordinates
- **Debug Mode**: Visualize AI's vision in real-time
- **Emergency Stop**: Move mouse to top-left corner to stop

## 🧠 Training Your Own AI Model

### Option 1: Use Simple Computer Vision (Default)
The AI uses color-based detection by default - no training required!

### Option 2: Train a Neural Network
```bash
python train_model.py
```

This will:
1. Help you collect training data by labeling moles vs holes
2. Train a PyTorch CNN model on your data
3. Save the trained model for the AI to use

## 📁 Project Structure

```
├── whack_a_mole.py          # Main game file
├── ai_game_player.py        # AI player that plays the game
├── mole_detection_model.py  # PyTorch CNN model for mole detection
├── train_model.py           # Training script for the neural network
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Requirements

### For the Game Only:
- Python 3.6+
- tkinter (included with Python by default)

### For the AI Player:
- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Pillow
- pyautogui
- mss (screen capture)
- matplotlib
- scikit-learn

## 🎮 Game Controls

### Human Mode:
- **START GAME**: Begin a new game
- **STOP GAME**: End the current game early
- **Mouse Click**: Whack the moles when they appear

### AI Mode:
- **Move mouse to corner**: Emergency stop
- **Menu options**: Control AI behavior
- **Q key**: Quit calibration mode

## ⚙️ Game Mechanics

- Moles appear for 1-3 seconds before disappearing
- New moles spawn every 0.5-1.5 seconds
- Only one mole appears at a time
- Missing a mole doesn't penalize your score
- The game automatically ends when time runs out

## 🔧 Troubleshooting

### AI Can't Find Game Window
1. Make sure the game window is fully visible
2. Try manual window selection
3. Adjust the color detection parameters in the code

### AI Clicks Wrong Positions
1. Use the calibration mode to see what the AI detects
2. Adjust the grid position estimation
3. Collect training data and train a custom model

### Performance Issues
1. Close other applications
2. Reduce the AI's frame rate in the code
3. Use a faster computer or GPU for neural network inference

## 🎯 AI Performance Tips

- **Best Results**: Use the color-based detection (default)
- **Training Data**: Collect 100+ samples of each class for good neural network performance
- **Window Position**: Keep the game window in the same position during AI play
- **Lighting**: Consistent screen brightness improves detection

Enjoy whacking moles with both human reflexes and artificial intelligence! 🐹💥🤖 