import numpy as np
import os
import argparse
from collections import deque
from tqdm import tqdm # Progress bar

# Import necessary components from other project folders
from training import config  # Import configuration variables
from training.logger import SimpleLogger
from rl_agent.agent import DQNAgent
from rl_agent.utils import preprocess_state, action_to_index
from othello_game.game import Board # Import the Board class
from othello_game.game import constants as game_consts # Import game constants

class OthelloEnv:
    """A wrapper around the Othello Board class to provide a gym-like interface."""
    def __init__(self):
        self.board = Board() # The core game logic
        self.current_player = game_consts.PLAYER_WHITE # White starts

    def reset(self):
        """Resets the environment to the starting state."""
        self.board = Board() # Create a new board instance
        self.current_player = game_consts.PLAYER_WHITE
        return self.board.grid.copy() # Return the initial board grid

    def get_state(self):
        """Returns the current board state."""
        return self.board.grid.copy()

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
            move_successful = self.board.make_move(action[0], action[1], self.current_player)
            if not move_successful:
                # This shouldn't happen if agent only chooses valid moves, but handle defensively
                print(f"Warning: Agent attempted invalid move {action} for player {self.current_player}")
                # Penalize invalid move attempt heavily? Or just end episode? Let's assign large negative reward and end.
                # This indicates a potential flaw in the agent's action masking or environment's valid move generation.
                return self.get_state(), -10.0, True, {"error": "Invalid move attempted"}
        else:
            # Handle pass move (action is None)
            # Check if passing was the only option
            if self.board.get_valid_moves(self.current_player):
                 print(f"Warning: Agent passed when valid moves exist for player {self.current_player}")
                 # Penalize passing when moves are available
                 return self.get_state(), -1.0, False, {"warning": "Passed with valid moves"}
            # else: pass was valid or forced

        # --- Switch Player and Check Game Status ---
        self.current_player *= -1 # Switch player
        player_valid_moves = self.board.get_valid_moves(self.current_player)
        opponent_valid_moves = self.board.get_valid_moves(-self.current_player)

        done = False
        reward = 0.0
        info = {}

        if not player_valid_moves and not opponent_valid_moves:
            # Game Over: Neither player has valid moves
            done = True
            white_score, black_score = self.board.get_score() # Get final score
            if white_score > black_score:
                winner = game_consts.PLAYER_WHITE
                reward = 1.0 if self.current_player == game_consts.PLAYER_WHITE else -1.0 # Reward relative to the player *whose turn it would have been*
            elif black_score > white_score:
                winner = game_consts.PLAYER_BLACK
                reward = 1.0 if self.current_player == game_consts.PLAYER_BLACK else -1.0
            else: # Draw
                winner = 0
                reward = 0.0 # Or a small positive reward for draw? Let's use 0.

            info = {"winner": winner, "white_score": white_score, "black_score": black_score}

        elif not player_valid_moves:
            # Current player has no moves, must pass - turn goes back to opponent
            # The environment state reflects the board *after* the original player's move,
            # but the 'current_player' is now the one who has to pass.
            # We return the state, 0 reward (game not over), and done=False.
            # The training loop needs to handle feeding the *same state* back to the agent
            # but for the *other* player in the next step.
            # Or, more simply, the step function can auto-pass here. Let's do that.
            # print(f"Player {self.current_player} must pass.")
            self.current_player *= -1 # Pass the turn back immediately
            # Reward is still 0 as the game isn't over

        return self.get_state(), reward, done, info


def train(args):
    """Main training function."""

    # --- Initialization ---
    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)
    logger = SimpleLogger(config.LOG_DIR, config.LOG_FILE)
    logger.log_message("Starting Othello DQN Training...")
    logger.log_message(f"Using config: {vars(config)}")
    logger.log_message(f"Arguments: {args}")

    env = OthelloEnv()
    agent = DQNAgent(
        state_shape=config.STATE_SHAPE,
        action_size=config.ACTION_SIZE,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        epsilon=config.EPSILON_START,
        epsilon_decay=config.EPSILON_DECAY,
        epsilon_min=config.EPSILON_MIN,
        buffer_size=config.BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        target_update_freq=config.TARGET_UPDATE_FREQ
    )

    if args.load_model:
        load_path = args.load_model
        if os.path.exists(load_path):
             logger.log_message(f"Loading model weights from: {load_path}")
             agent.load(load_path)
             # Optionally reset epsilon if continuing training from a saved state
             # agent.epsilon = max(config.EPSILON_MIN, some_saved_epsilon)
        else:
            logger.log_message(f"Warning: Load model path not found: {load_path}. Starting fresh.")


    total_steps = 0
    episode_rewards = deque(maxlen=100) # Store rewards of last 100 episodes for avg

    # --- Training Loop ---
    for episode in tqdm(range(1, config.NUM_EPISODES + 1), desc="Training Episodes"):
        state_grid = env.reset()
        current_player = game_consts.PLAYER_WHITE # White starts
        processed_state = preprocess_state(state_grid, current_player)
        episode_reward = 0
        episode_steps = 0
        done = False
        game_result = "In Progress"

        while not done and episode_steps < config.MAX_STEPS_PER_EPISODE:
            valid_moves = env.get_valid_moves(current_player)

            # Agent selects action
            if not valid_moves:
                # Player must pass
                action = None
                action_index = action_to_index(action) # Get index for pass
            else:
                action, action_index = agent.select_action(processed_state, valid_moves)

            # Environment takes step
            # Note: env.step handles the turn switch internally
            next_state_grid, reward, done, info = env.step(action)
            next_player = env.current_player # Player whose turn it is *now*
            processed_next_state = preprocess_state(next_state_grid, next_player)

            # Store experience (state is from POV of player who *made* the move)
            agent.store_experience(processed_state, action_index, reward, processed_next_state, done)

            # Agent learns
            agent.learn()

            # Update state and metrics
            processed_state = processed_next_state
            current_player = next_player # Update player for the next loop iteration
            episode_reward += reward # Accumulate reward (usually 0 until the end)
            episode_steps += 1
            total_steps += 1

            # Log step info periodically
            if total_steps % config.LOG_FREQ_STEP == 0:
                 logger.log_step(total_steps, {'epsilon': agent.epsilon})


            if done:
                if "winner" in info:
                    winner = info['winner']
                    game_result = "Draw" if winner == 0 else ("White Wins" if winner == game_consts.PLAYER_WHITE else "Black Wins")
                else:
                    game_result = info.get("error", "Ended") # Game ended due to error or max steps

        # --- End of Episode ---
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)

        if episode % config.LOG_FREQ_EPISODE == 0:
             logger.log_episode(episode, total_steps, {
                 'reward': episode_reward,
                 'avg_reward_100': avg_reward,
                 'steps': episode_steps,
                 'epsilon': agent.epsilon,
                 'result': game_result
             })

        # Save model periodically
        if episode % config.SAVE_FREQ == 0:
            save_path = os.path.join(config.MODEL_SAVE_DIR, f"othello_dqn_ep_{episode}.h5")
            agent.save(save_path)
            # Optionally save a consistent latest model file
            latest_save_path = os.path.join(config.MODEL_SAVE_DIR, "othello_dqn_latest.h5")
            agent.save(latest_save_path)

    # --- End of Training ---
    logger.log_message("Training Finished.")
    final_save_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)
    agent.save(final_save_path)
    logger.log_message(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN Agent for Othello.")
    parser.add_argument("--load_model", type=str, default=config.LOAD_MODEL_PATH,
                        help="Path to load pre-trained model weights from.")
    # Add more arguments if needed, e.g., --episodes, --learning_rate etc.
    # to override config values via command line. Example:
    # parser.add_argument("--episodes", type=int, default=config.NUM_EPISODES,
    #                     help="Number of episodes to train for.")

    args = parser.parse_args()

    # Example of overriding config from args if argument is provided
    # if args.episodes is not None:
    #    config.NUM_EPISODES = args.episodes

    train(args)