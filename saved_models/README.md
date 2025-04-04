# Saved Models

This directory stores the trained model weights for the Othello Reinforcement Learning agent.

## Naming Convention

A suggested naming convention for saved models during training is:

`othello_dqn_ep_<episode_number>.h5`

The final trained model might be saved as:

`othello_dqn_agent.h5` or `othello_dqn_latest.h5`

Models saved here are typically Keras/TensorFlow `.h5` files containing the weights of the DQN agent's neural network. You can load these weights using the `--load_model` argument in `training/train_agent.py` to resume training or use the agent for evaluation/play.