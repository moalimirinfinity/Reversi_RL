<<<<<<< HEAD
# ♟️ Othello AI with Reinforcement Learning

This project implements an **Othello (Reversi) game** with a **Deep Q-Networks (DQN) Reinforcement Learning (RL) agent**. The agent learns to play the game through **self-play training**, improving over time.

## 🚀 Features

- ✅ **Interactive Othello Game** – Play Othello via a **Pygame UI**  
- ✅ **DQN-Based AI** – A Reinforcement Learning agent built with **TensorFlow/Keras**  
- ✅ **Self-Play Training** – Train the AI by making it play against itself  
- ✅ **Configurable Training** – Modify hyperparameters in a **central config file**  
- ✅ **Model Persistence** – Save and load models to resume training or play against AI  

---

## 📂 Project Structure

```bash
Othello_RL/
│
├── othello_game/        # Othello game logic & UI (Pygame)
│   ├── game/
│   │   ├── __init__.py
│   │   ├── board.py       # Board logic
│   │   ├── constants.py   # Game constants
│   │   └── game.py        # Game management
│   ├── __init__.py
│   └── main.py           # Run this to play the game
│
├── rl_agent/            # DQN-based RL agent
│   ├── __init__.py
│   ├── agent.py         # DQNAgent class
│   ├── models.py        # Neural network models (Keras)
│   ├── replay_buffer.py # Experience replay buffer
│   └── utils.py         # State preprocessing, action mapping
│
├── training/            # Training scripts & config
│   ├── __init__.py
│   ├── config.py        # Hyperparameters & settings
│   ├── logger.py        # Logging utility
│   └── train_agent.py   # Run this to train the agent
│
├── saved_models/        # Trained model weights
│   └── README.md
│
├── data/                # Logs, training results
│   └── README.md
│
├── requirements.txt     # Dependencies
└── README.md            # This file

=======
# Othello AI with Reinforcement Learning

This project implements an Othello (Reversi) game where a Reinforcement Learning agent, specifically using Deep Q-Networks (DQN), learns to play the game by playing against itself.

## Features

* **Playable Othello Game:** An interactive Othello game built with Pygame.
* **DQN Agent:** A Reinforcement Learning agent implemented using TensorFlow/Keras.
* **Self-Play Training:** A script to train the agent by having it play against itself.
* **Configurable Training:** Hyperparameters and settings managed via a central configuration file.
* **Model Saving/Loading:** Ability to save training progress and load models to resume or play.

## Project Structure
>>>>>>> a6ce5ef (the agents added)
