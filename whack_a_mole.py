import tkinter as tk
from tkinter import messagebox
import random
import threading
import time

class WhackAMoleGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Whack-A-Mole Game")
        self.root.geometry("500x600")
        self.root.configure(bg='#2E8B57')  # Forest green background
        
        # Game variables
        self.score = 0
        self.time_left = 30  # 30 seconds game
        self.game_running = False
        self.mole_buttons = []
        self.current_mole = None
        self.mole_timer = None
        
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="🔨 WHACK-A-MOLE 🔨", 
            font=("Arial", 24, "bold"),
            bg='#2E8B57',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Score and time frame
        info_frame = tk.Frame(self.root, bg='#2E8B57')
        info_frame.pack(pady=10)
        
        self.score_label = tk.Label(
            info_frame,
            text=f"Score: {self.score}",
            font=("Arial", 16, "bold"),
            bg='#2E8B57',
            fg='white'
        )
        self.score_label.pack(side=tk.LEFT, padx=20)
        
        self.time_label = tk.Label(
            info_frame,
            text=f"Time: {self.time_left}",
            font=("Arial", 16, "bold"),
            bg='#2E8B57',
            fg='white'
        )
        self.time_label.pack(side=tk.RIGHT, padx=20)
        
        # Game grid frame
        self.game_frame = tk.Frame(self.root, bg='#2E8B57')
        self.game_frame.pack(pady=20)
        
        # Create 3x3 grid of holes
        for i in range(3):
            row = []
            for j in range(3):
                button = tk.Button(
                    self.game_frame,
                    text="🕳️",  # Hole emoji
                    font=("Arial", 36),
                    width=4,
                    height=2,
                    command=lambda r=i, c=j: self.whack_mole(r, c),
                    bg='#8B4513',  # Brown color for holes
                    relief='sunken'
                )
                button.grid(row=i, column=j, padx=5, pady=5)
                row.append(button)
            self.mole_buttons.append(row)
        
        # Control buttons frame
        control_frame = tk.Frame(self.root, bg='#2E8B57')
        control_frame.pack(pady=20)
        
        self.start_button = tk.Button(
            control_frame,
            text="START GAME",
            font=("Arial", 14, "bold"),
            command=self.start_game,
            bg='#32CD32',
            fg='white',
            padx=20,
            pady=10
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = tk.Button(
            control_frame,
            text="STOP GAME",
            font=("Arial", 14, "bold"),
            command=self.stop_game,
            bg='#DC143C',
            fg='white',
            padx=20,
            pady=10,
            state='disabled'
        )
        self.stop_button.pack(side=tk.RIGHT, padx=10)
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="Click START GAME, then whack the moles (🐹) when they appear!\nYou have 30 seconds to get the highest score!",
            font=("Arial", 12),
            bg='#2E8B57',
            fg='white',
            justify=tk.CENTER
        )
        instructions.pack(pady=10)
    
    def start_game(self):
        """Start the whack-a-mole game"""
        self.game_running = True
        self.score = 0
        self.time_left = 30
        self.update_display()
        
        # Disable start button, enable stop button
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Reset all holes
        self.hide_all_moles()
        
        # Start the game timer
        self.start_timer()
        
        # Start spawning moles
        self.spawn_mole()
    
    def stop_game(self):
        """Stop the game"""
        self.game_running = False
        
        # Enable start button, disable stop button
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        # Hide current mole
        self.hide_all_moles()
        
        # Cancel mole timer
        if self.mole_timer:
            self.root.after_cancel(self.mole_timer)
        
        # Show final score
        messagebox.showinfo("Game Over", f"Final Score: {self.score}")
    
    def start_timer(self):
        """Start the game countdown timer"""
        if self.game_running and self.time_left > 0:
            self.time_left -= 1
            self.update_display()
            self.root.after(1000, self.start_timer)  # Call again after 1 second
        elif self.time_left <= 0:
            self.stop_game()
    
    def spawn_mole(self):
        """Randomly spawn a mole in one of the holes"""
        if not self.game_running:
            return
            
        # Hide current mole
        self.hide_all_moles()
        
        # Random position for new mole
        row = random.randint(0, 2)
        col = random.randint(0, 2)
        
        # Show mole
        self.current_mole = (row, col)
        self.mole_buttons[row][col].config(text="🐹", bg='#FFD700')  # Golden mole
        
        # Hide mole after random time (1-3 seconds)
        hide_time = random.randint(1000, 3000)  # milliseconds
        self.mole_timer = self.root.after(hide_time, self.hide_current_mole)
        
        # Schedule next mole spawn
        spawn_time = random.randint(500, 1500)  # milliseconds
        self.root.after(spawn_time, self.spawn_mole)
    
    def hide_current_mole(self):
        """Hide the current mole"""
        if self.current_mole:
            row, col = self.current_mole
            self.mole_buttons[row][col].config(text="🕳️", bg='#8B4513')
            self.current_mole = None
    
    def hide_all_moles(self):
        """Hide all moles and reset holes"""
        for i in range(3):
            for j in range(3):
                self.mole_buttons[i][j].config(text="🕳️", bg='#8B4513')
        self.current_mole = None
    
    def whack_mole(self, row, col):
        """Handle mole whacking"""
        if not self.game_running:
            return
            
        # Check if there's a mole at this position
        if self.current_mole and self.current_mole == (row, col):
            # Successful whack!
            self.score += 10
            self.update_display()
            
            # Visual feedback
            self.mole_buttons[row][col].config(text="💥", bg='#FF4500')
            self.root.after(200, lambda: self.mole_buttons[row][col].config(text="🕳️", bg='#8B4513'))
            
            # Hide the mole
            self.current_mole = None
            
            # Cancel the hide timer since mole was whacked
            if self.mole_timer:
                self.root.after_cancel(self.mole_timer)
    
    def update_display(self):
        """Update score and time display"""
        self.score_label.config(text=f"Score: {self.score}")
        self.time_label.config(text=f"Time: {self.time_left}")
    
    def on_closing(self):
        """Handle window closing"""
        self.game_running = False
        self.root.destroy()
    
    def run(self):
        """Start the game application"""
        self.root.mainloop()

if __name__ == "__main__":
    game = WhackAMoleGame()
    game.run() 