import numpy as np
# Assuming BOARD_SIZE is available, e.g., imported from othello_game.game.constants
# If not, define it here or pass it as an argument.
from othello_game.game.constants import BOARD_SIZE # Example import

# BOARD_SIZE = 8 # Define or import appropriately - REMOVED

def preprocess_state(board_grid, player):
    """
    Converts the board state into a format suitable for the neural network.

    Args:
        board_grid (np.ndarray): The 8x8 board grid (-1 for black, 1 for white, 0 for empty).
        player (int): The current player (1 for white, -1 for black).

    Returns:
        np.ndarray: A processed state representation (e.g., 3x8x8).
                    Channel 0: Current player's pieces (1) vs empty/opponent (0)
                    Channel 1: Opponent's pieces (1) vs empty/current (0)
                    Channel 2: All 1s if player is White (1), all -1s if player is Black (-1) - encodes turn
    """
    state = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Channel 0: Player's pieces
    state[0] = (board_grid == player).astype(np.float32)

    # Channel 1: Opponent's pieces
    state[1] = (board_grid == -player).astype(np.float32)

    # Channel 2: Player indicator (uniform plane)
    state[2] = np.full((BOARD_SIZE, BOARD_SIZE), player, dtype=np.float32)

    # Reshape for Keras CNN if needed (channels_last format: height, width, channels)
    state = np.transpose(state, (1, 2, 0)) # Convert to (8, 8, 3)

    return state

def action_to_index(action):
    """Converts a (row, col) action to a flat index."""
    if action is None: # Handle pass move if applicable
        return BOARD_SIZE * BOARD_SIZE # Use index 64 for pass
    row, col = action
    return row * BOARD_SIZE + col

def index_to_action(index):
    """Converts a flat index back to a (row, col) action."""
    if index == BOARD_SIZE * BOARD_SIZE: # Handle pass move index
        return None
    row = index // BOARD_SIZE
    col = index % BOARD_SIZE
    return (row, col)