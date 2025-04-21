import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import BOARD_SIZE from game constants
from othello_game.game.constants import BOARD_SIZE

# Assuming BOARD_SIZE is available
# BOARD_SIZE = 8 # Define or import appropriately - REMOVED
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1 # 64 possible move locations + 1 for pass

def create_dqn_model(input_shape, num_actions=NUM_ACTIONS):
    """
    Creates a Convolutional Neural Network model for the DQN agent.

    Args:
        input_shape (tuple): The shape of the preprocessed state (e.g., (8, 8, 3)).
        num_actions (int): The number of possible outputs (Q-values for each action).

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_actions, activation="linear"),  # Linear activation for Q-values
        ]
    )
    return model