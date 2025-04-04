import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

from .models import create_dqn_model
from .replay_buffer import ReplayBuffer
from .utils import preprocess_state, action_to_index, index_to_action

# Assuming BOARD_SIZE and other constants are available
BOARD_SIZE = 8 # Define or import appropriately
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE

class DQNAgent:
    def __init__(self, state_shape, action_size=NUM_ACTIONS,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.9995, epsilon_min=0.01,
                 buffer_size=100000, batch_size=64,
                 target_update_freq=100):
        """
        Initialize the DQN Agent.

        Args:
            state_shape (tuple): Shape of the preprocessed state input (e.g., (8, 8, 3)).
            action_size (int): Number of possible actions.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Rate at which epsilon decreases.
            epsilon_min (float): Minimum value for epsilon.
            buffer_size (int): Max size of the replay buffer.
            batch_size (int): Size of the batches sampled from the buffer for training.
            target_update_freq (int): How often (in steps) to update the target network.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0 # For periodic target network updates

        # Create main model and target model
        self.model = create_dqn_model(state_shape, action_size)
        self.target_model = create_dqn_model(state_shape, action_size)
        self.target_model.set_weights(self.model.get_weights()) # Initialize target same as main

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_function = keras.losses.MeanSquaredError()

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, valid_moves):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): The current preprocessed game state.
            valid_moves (list): List of valid (row, col) moves.

        Returns:
            tuple: The chosen action (row, col) or None for pass.
            int: The index of the chosen action.
        """
        if not valid_moves: # Handle case where there are no valid moves (pass)
            return None, action_to_index(None) # Return None action and pass index

        if np.random.rand() <= self.epsilon:
            # Exploration: choose a random valid move
            action = random.choice(valid_moves)
            action_index = action_to_index(action)
            return action, action_index
        else:
            # Exploitation: choose the best action predicted by the model among valid moves
            state_tensor = tf.convert_to_tensor([state]) # Add batch dimension
            q_values = self.model(state_tensor, training=False)[0].numpy() # Get Q-values

            # Mask invalid actions by setting their Q-values very low
            valid_action_indices = [action_to_index(move) for move in valid_moves]
            masked_q_values = np.full(self.action_size, -np.inf) # Start with all invalid
            masked_q_values[valid_action_indices] = q_values[valid_action_indices] # Set valid Q's

            best_action_index = np.argmax(masked_q_values)
            action = index_to_action(best_action_index)
            return action, best_action_index

    def store_experience(self, state, action_index, reward, next_state, done):
        """Stores a transition in the replay buffer."""
        self.replay_buffer.add(state, action_index, reward, next_state, done)

    def learn(self):
        """Trains the main network using a batch sampled from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return # Not enough experiences yet

        # Sample a batch
        states, action_indices, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        action_indices = tf.convert_to_tensor(action_indices, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32) # Use float for masking

        # Predict Q-values for the next states using the target network
        # For DQN: Q_target = r + gamma * max_a'( Q_target(s', a') )
        future_rewards = self.target_model(next_states, training=False)
        # Use max Q-value for non-terminal next states
        updated_q_values = rewards + self.gamma * tf.reduce_max(future_rewards, axis=1) * (1 - dones)

        # Create a mask to only update the Q-value for the action taken
        masks = tf.one_hot(action_indices, self.action_size)

        with tf.GradientTape() as tape:
            # Predict Q-values for the current states using the main network
            q_values = self.model(states)
            # Select the Q-value for the action that was actually taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between predicted Q-value and the target Q-value
            loss = self.loss_function(updated_q_values, q_action)

        # Backpropagate loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon) # Ensure it doesn't go below min

        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        """Copies weights from the main model to the target model."""
        print("Updating target network...")
        self.target_model.set_weights(self.model.get_weights())

    def save(self, filepath):
        """Saves the model weights."""
        self.model.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    def load(self, filepath):
        """Loads the model weights."""
        try:
            self.model.load_weights(filepath)
            self.target_model.load_weights(filepath) # Keep target aligned after loading
            print(f"Model weights loaded from {filepath}")
        except Exception as e:
            print(f"Error loading weights from {filepath}: {e}. Ensure the model architecture matches.")
