
import numpy as np
import os
import argparse
from collections import deque
from tqdm import tqdm # Progress bar

# Import necessary components using relative paths suitable for running as a module
# (e.g., python -m training.train_agent from the Othello_RL directory)
from . import config  # Import configuration variables
from .logger import SimpleLogger # Import logger
from ..rl_agent.agent import DQNAgent # Import the agent
from ..rl_agent.utils import preprocess_state, action_to_index # Import RL utilities
from ..othello_game.game import Board # Import the Board class
from ..othello_game.game import constants as game_consts # Import game constants

class OthelloEnv:
    """A wrapper around the Othello Board class to provide a gym-like interface."""
    def __init__(self):
        self.board = Board() # The core game logic
        self.current_player = game_consts.PLAYER_WHITE # White starts

    def reset(self):
        """Resets the environment to the starting state."""
        self.board = Board() # Create a new board instance
        self.current_player = game_consts.PLAYER_WHITE # Reset player
        return self.board.grid.copy() # Return the initial board grid

    def get_state(self):
        """Returns the current board state."""
        return self.board.grid.copy() # Return grid copy

    def get_valid_moves(self, player):
        """Returns a list of valid (row, col) moves for the given player."""
        return self.board.get_valid_moves(player) # Use Board's method

    def step(self, action):
        """
        Applies an action (move) to the environment.

        Args:
            action (tuple): The (row, col) of the move, or None for pass.

        Returns:
            tuple: (next_state, reward, done, info)
                   next_state (np.ndarray): The board state after the move.
                   reward (float): The reward obtained.
                   done (bool): Whether the episode has ended.
                   info (dict): Additional info (e.g., winner).
        """
        # --- Apply action ---
        if action is not None:
            move_successful = self.board.make_move(action[0], action[1], self.current_player) # Apply move
            if not move_successful:
                # Handle defensively - should not happen if agent masks correctly
                print(f"Warning: Agent attempted invalid move {action} for player {self.current_player}")
                # Penalize invalid move attempt heavily and end episode
                return self.get_state(), -10.0, True, {"error": "Invalid move attempted"}
        else:
            # Handle pass move (action is None)
            # Check if passing was the only option
            if self.board.get_valid_moves(self.current_player): # Check if valid moves existed
                 print(f"Warning: Agent passed when valid moves exist for player {self.current_player}")
                 # Penalize passing when moves are available
                 return self.get_state(), -1.0, False, {"warning": "Passed with valid moves"}
            # else: pass was valid or forced

        # --- Switch Player and Check Game Status ---
        self.current_player *= -1 # Switch player
        player_valid_moves = self.board.get_valid_moves(self.current_player) # Get moves for new current player
        opponent_valid_moves = self.board.get_valid_moves(-self.current_player) # Get moves for opponent

        done = False
        reward = 0.0
        info = {}

        if not player_valid_moves and not opponent_valid_moves:
            # Game Over: Neither player has valid moves
            done = True
            white_score, black_score = self.board.get_score() # Get final score
            winner = 0
            if white_score > black_score:
                winner = game_consts.PLAYER_WHITE # White wins
            elif black_score > white_score:
                winner = game_consts.PLAYER_BLACK # Black wins
            else: # Draw
                winner = 0

            # Assign reward based on the outcome for the player who *just* acted ('-self.current_player').
            acting_player = -self.current_player
            if winner == acting_player:
                reward = 1.0 # Player who just acted won
            elif winner == -acting_player:
                reward = -1.0 # Player who just acted lost
            else: # Draw
                reward = 0.0 # Draw

            info = {"winner": winner, "white_score": white_score, "black_score": black_score}

        elif not player_valid_moves:
            # Current player has no moves, must pass - turn goes back to opponent immediately.
            # print(f"Player {self.current_player} must pass.")
            self.current_player *= -1 # Pass the turn back immediately
            # Reward is still 0 as the game isn't over

        return self.get_state(), reward, done, info


def train(args):
    """Main training function."""

    # --- Initialization ---
    if not os.path.exists(config.MODEL_SAVE_DIR): # Check save dir
        os.makedirs(config.MODEL_SAVE_DIR) # Create save dir
    logger = SimpleLogger(config.LOG_DIR, config.LOG_FILE) # Init logger
    logger.log_message("Starting Othello DQN Training...") # Log start
    logger.log_message(f"Using config: {vars(config)}") # Log config
    logger.log_message(f"Arguments: {args}") # Log args

    env = OthelloEnv() # Create environment
    agent = DQNAgent(
        state_shape=config.STATE_SHAPE, # Agent state shape
        action_size=config.ACTION_SIZE, # Agent action size
        learning_rate=config.LEARNING_RATE, # Agent learning rate
        gamma=config.GAMMA, # Agent discount factor
        epsilon=config.EPSILON_START, # Agent initial epsilon
        epsilon_decay=config.EPSILON_DECAY, # Agent epsilon decay
        epsilon_min=config.EPSILON_MIN, # Agent min epsilon
        buffer_size=config.BUFFER_SIZE, # Agent buffer size
        batch_size=config.BATCH_SIZE, # Agent batch size
        target_update_freq=config.TARGET_UPDATE_FREQ # Agent target update freq
    ) # Create agent instance

    # Load model weights if specified
    load_path = args.load_model if args.load_model else config.LOAD_MODEL_PATH # Determine load path
    if load_path and os.path.exists(load_path):
         logger.log_message(f"Loading model weights from: {load_path}") # Log loading
         agent.load(load_path) # Load weights
         # Optionally reset epsilon if continuing training, e.g., agent.epsilon = config.EPSILON_MIN
    elif args.load_model: # Only warn if the user specified a path that doesn't exist
        logger.log_message(f"Warning: Load model path specified but not found: {load_path}. Starting fresh.") # Log warning


    total_steps = 0
    episode_rewards = deque(maxlen=100) # Store rewards of last 100 episodes for avg

    # --- Training Loop ---
    for episode in tqdm(range(1, config.NUM_EPISODES + 1), desc="Training Episodes"): # Loop through episodes
        state_grid = env.reset() # Reset environment
        current_player = game_consts.PLAYER_WHITE # White starts
        processed_state = preprocess_state(state_grid, current_player) # Preprocess initial state
        episode_reward = 0
        episode_steps = 0
        done = False
        game_result = "In Progress"

        while not done and episode_steps < config.MAX_STEPS_PER_EPISODE: # Loop within episode
            valid_moves = env.get_valid_moves(current_player) # Get valid moves

            # Agent selects action
            if not valid_moves:
                # Player must pass
                action = None
                action_index = action_to_index(action) # Get index for pass (64)
            else:
                # Agent selects a valid move using its policy
                action, action_index = agent.select_action(processed_state, valid_moves) # Select action

            # Environment takes step
            # Note: env.step handles the turn switch internally
            next_state_grid, reward, done, info = env.step(action) # Perform step in env
            next_player = env.current_player # Player whose turn it is *now*
            processed_next_state = preprocess_state(next_state_grid, next_player) # Preprocess next state

            # Store experience only if the agent made a move (not a forced pass handled by env)
            # The reward is associated with the state and action that led to it.
            # The 'done' flag indicates if this action ended the game.
            agent.store_experience(processed_state, action_index, reward, processed_next_state, done) # Store experience

            # Agent learns from stored experiences
            agent.learn() # Trigger learning step

            # Update state and metrics
            processed_state = processed_next_state # Move to next state
            current_player = next_player # Update player for the next loop iteration
            episode_reward += reward # Accumulate reward
            episode_steps += 1
            total_steps += 1

            # Log step info periodically
            if total_steps % config.LOG_FREQ_STEP == 0: # Check log frequency
                 logger.log_step(total_steps, {'epsilon': agent.epsilon}) # Log step info

            # Check if game ended in this step
            if done:
                if "winner" in info:
                    winner = info['winner']
                    game_result = "Draw" if winner == 0 else ("White Wins" if winner == game_consts.PLAYER_WHITE else "Black Wins") # Determine result
                elif "error" in info:
                     game_result = info["error"] # Game ended due to error
                elif episode_steps >= config.MAX_STEPS_PER_EPISODE:
                     game_result = f"Ended (Max Steps {config.MAX_STEPS_PER_EPISODE})" # Game ended due to max steps
                else:
                     game_result = "Ended (Unknown)"

        # --- End of Episode ---
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0

        # Log episode summary periodically
        if episode % config.LOG_FREQ_EPISODE == 0: # Check log frequency
             logger.log_episode(episode, total_steps, {
                 'reward': episode_reward,
                 'avg_reward_100': avg_reward,
                 'steps': episode_steps,
                 'epsilon': agent.epsilon, # Current epsilon
                 'result': game_result
             }) # Log episode stats

        # Save model periodically
        if episode % config.SAVE_FREQ == 0: # Check save frequency
            save_path = os.path.join(config.MODEL_SAVE_DIR, f"othello_dqn_ep_{episode}.h5") # Define save path
            agent.save(save_path) # Save model
            # Optionally save a consistent latest model file
            latest_save_path = os.path.join(config.MODEL_SAVE_DIR, "othello_dqn_latest.h5") # Define latest save path
            agent.save(latest_save_path) # Save latest model

    # --- End of Training ---
    logger.log_message("Training Finished.") # Log finish
    final_save_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME) # Define final path
    agent.save(final_save_path) # Save final model
    logger.log_message(f"Final model saved to {final_save_path}") # Log final save path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN Agent for Othello.")
    # Use default=None for load_model to distinguish between not provided and explicitly set to None in config
    parser.add_argument("--load_model", type=str, default=None,
                        help=f"Path to load pre-trained model weights (overrides config value: {config.LOAD_MODEL_PATH}).") #
    # Add arguments to potentially override config values via command line
    # Example:
    # parser.add_argument("--episodes", type=int, default=None,
    #                     help=f"Number of episodes to train for (overrides config value: {config.NUM_EPISODES}).") #
    # parser.add_argument("--lr", type=float, default=None,
    #                     help=f"Learning rate (overrides config value: {config.LEARNING_RATE}).") #

    args = parser.parse_args()

    # Example of overriding config from args if argument is provided
    # if args.episodes is not None:
    #    config.NUM_EPISODES = args.episodes
    # if args.lr is not None:
    #    config.LEARNING_RATE = args.lr
    # (You would need to re-initialize the agent if LR changes after creation,
    #  so it's better to pass args to the train function or agent constructor)

    train(args) # Start training