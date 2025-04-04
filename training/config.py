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
MODEL_SAVE_DIR = "../saved_models/" # Directory to save trained models
MODEL_NAME = "othello_dqn_agent.h5" # Base name for saved model weights
SAVE_FREQ = 200                # Save the model every N episodes

LOAD_MODEL_PATH = None         # Set to a file path (e.g., "../saved_models/othello_dqn_agent.h5") to load weights and continue training
                               # Set to None to start training from scratch

# --- Logging ---
LOG_DIR = "../data/logs/"      # Directory for log files
LOG_FILE = "training_log.txt"  # Name of the log file
LOG_FREQ_EPISODE = 10          # Log summary statistics every N episodes
LOG_FREQ_STEP = 1000           # Log step-wise info (like epsilon) every N steps

# --- Game Environment ---
BOARD_SIZE = 8                 # Should match the constant in othello_game

# --- Agent Architecture ---
# Assuming the state shape is (BOARD_SIZE, BOARD_SIZE, 3) based on rl_agent.utils.preprocess_state
STATE_SHAPE = (BOARD_SIZE, BOARD_SIZE, 3)
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE # 64 possible moves