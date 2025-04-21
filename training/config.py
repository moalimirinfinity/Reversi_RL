import os # Keep os for environment variables if needed in future
from pathlib import Path

# --- Base Path ---
# Assumes config.py is in the 'training' directory
# PROJECT_ROOT = Path(__file__).resolve().parent.parent # Get project root (one level up from 'training')
# Simpler: Define relative to script location is often sufficient if run structure is consistent
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent

# Configuration settings for training the Othello RL agent

# --- RL Hyperparameters ---
LEARNING_RATE = 0.0005          # Learning rate for Adam optimizer
GAMMA = 0.99                   # Discount factor for future rewards
BUFFER_SIZE = 50000            # Max size of the experience replay buffer
BATCH_SIZE = 64                # Batch size for sampling from the replay buffer
TARGET_UPDATE_FREQ = 500      # How often (in agent steps) to update the target network

# --- Epsilon Greedy Strategy (Exploration) ---
EPSILON_START = 1.0            # Initial epsilon value
EPSILON_DECAY = 0.999          # Multiplicative decay factor for epsilon per episode/step
EPSILON_MIN = 0.05             # Minimum epsilon value

# --- Training Parameters ---
NUM_EPISODES = 10000           # Total number of games (episodes) to train for
MAX_STEPS_PER_EPISODE = 100    # Maximum steps allowed per game (prevents infinite loops in case of bugs)

# --- Saving/Loading Models ---
# Define paths relative to project root
MODEL_SAVE_DIR = PROJECT_ROOT / "saved_models" # Directory to save trained models
MODEL_NAME = "othello_dqn_agent.h5" # Base name for saved model weights
SAVE_FREQ = 200                # Save the model every N episodes

# Default path to load from (can be None)
DEFAULT_LOAD_PATH = MODEL_SAVE_DIR / MODEL_NAME # Default load path based on save dir/name
LOAD_MODEL_PATH = None         # Set to a file path string or Path object to load weights and continue training
                               # Set to None to start training from scratch. Can be overridden by command line args.

# --- Logging ---
# Define paths relative to project root
LOG_DIR = PROJECT_ROOT / "data" / "logs"     # Directory for log files
LOG_FILE = "training_log.txt"  # Name of the log file
LOG_FREQ_EPISODE = 10          # Log summary statistics every N episodes
LOG_FREQ_STEP = 1000           # Log step-wise info (like epsilon) every N steps

# --- Game Environment ---
# Import BOARD_SIZE from the game constants
from othello_game.game.constants import BOARD_SIZE
# BOARD_SIZE = 8                 # Should match the constant in othello_game - REMOVED

# --- Agent Architecture ---
# Assuming the state shape is (BOARD_SIZE, BOARD_SIZE, 3) based on rl_agent.utils.preprocess_state
STATE_SHAPE = (BOARD_SIZE, BOARD_SIZE, 3)
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE + 1 # 64 possible moves + 1 for pass