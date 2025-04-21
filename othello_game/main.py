import argparse
import os
import sys

# Adjust path to import from parent directory (if running main.py directly)
# This allows finding rl_agent and training modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from othello_game.game.game import OthelloGame
from rl_agent.agent import DQNAgent
from training import config # Import config to get defaults


def main():
    parser = argparse.ArgumentParser(description="Play Othello Human vs Human or Human vs AI.")
    parser.add_argument("--white", default="human", choices=["human", "ai"],
                        help="Player type for White (human or ai)")
    parser.add_argument("--black", default="human", choices=["human", "ai"],
                        help="Player type for Black (human or ai)")
    parser.add_argument("--model_path", type=str, default=str(config.DEFAULT_LOAD_PATH),
                        help=f"Path to load DQN model weights if playing against AI. Defaults to '{config.DEFAULT_LOAD_PATH}'.")

    args = parser.parse_args()

    agent_white = None
    agent_black = None
    ai_model_loaded = False

    # Load AI model if needed
    if args.white == "ai" or args.black == "ai":
        model_path = args.model_path
        if not model_path or not os.path.exists(model_path):
            print(f"Error: AI opponent selected, but model file not found at '{model_path}'.")
            print("Please train a model first (python -m training.train_agent) or provide a valid --model_path.")
            sys.exit(1)

        print(f"Loading AI model from {model_path}...")
        # Create a dummy agent instance to load weights into
        # Use settings from config for consistency
        temp_agent = DQNAgent(state_shape=config.STATE_SHAPE,
                              action_size=config.ACTION_SIZE)
        try:
            temp_agent.load(model_path)
            ai_model_loaded = True
            print("AI Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights from {model_path}: {e}")
            print("Ensure the model file corresponds to the current agent architecture.")
            sys.exit(1)

        # Assign the loaded agent to the correct player
        if args.white == "ai":
            agent_white = temp_agent
        if args.black == "ai":
            # If both are AI, they share the same loaded model instance
            agent_black = temp_agent

    # Create and run the game
    print(f"Starting game: White ({args.white}) vs Black ({args.black})")
    game = OthelloGame(agent_white=agent_white, agent_black=agent_black)
    game.run()


if __name__ == "__main__":
    main()