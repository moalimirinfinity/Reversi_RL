# Othello AI with Reinforcement Learning

This project implements an **Othello (Reversi) game** playable via a Pygame UI, featuring a **Deep Q-Networks (DQN) agent** trained using reinforcement learning (self-play). Play Human vs Human, Human vs AI, or watch AI vs AI!

## Features

- ✅ **Interactive Othello Game:** Play via a Pygame UI.
- ✅ **Multiple Modes:** Human vs Human, Human vs AI, AI vs AI via command-line options.
- ✅ **DQN Agent:** RL agent built with TensorFlow/Keras.
- ✅ **Self-Play Training:** Includes a script to train the agent.
- ✅ **Configurable:** Training hyperparameters and paths are centralized.
- ✅ **Model Persistence:** Save/load trained models.

---

## ⚙️ Setup

1.  **Clone:** `git clone <your-repo-url> && cd Othello_RL`
2.  **Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate | macOS/Linux: source venv/bin/activate
    ```
3.  **Install:** `pip install -r requirements.txt`
    *(Requires Python 3.x, Pygame, NumPy, TensorFlow)*

---

## ▶️ How to Run

### Training the AI

Train the agent using settings from `training/config.py`:

```bash
python -m training.train_agent
```

- Models are saved to `saved_models/`, logs to `data/logs/`.
- Resume training: `python -m training.train_agent --load_model path/to/your_model.h5`

### Playing the Game

Run the game interface:

```bash
python othello_game/main.py [options]
```

**Options:**

-   `--white <type>`: `human` or `ai` (Default: `human`)
-   `--black <type>`: `human` or `ai` (Default: `human`)
-   `--model_path <path>`: Path to `.h5` model for AI players (Defaults to `saved_models/othello_dqn_agent.h5`).

**Examples:**

-   **Human vs Human:** `python othello_game/main.py`
-   **Human vs AI:** `python othello_game/main.py --black ai [--model_path path/to/model.h5]`
-   **AI vs AI:** `python othello_game/main.py --white ai --black ai [--model_path path/to/model.h5]`

---

## 📂 Project Structure

```
Othello_RL/
├── othello_game/     # Game logic & UI (Pygame)
│   ├── game/         # Core game classes (Board, Game, Constants)
│   └── main.py       # Entry point to PLAY (handles args)
├── rl_agent/         # DQN Agent implementation
│   ├── agent.py      # DQNAgent class (select_action, learn, load/save)
│   ├── models.py     # Keras model definition
│   ├── replay_buffer.py
│   └── utils.py      # State preprocessing, action mapping
├── training/         # Agent training scripts
│   ├── config.py     # Hyperparameters, paths, settings
│   ├── logger.py
│   └── train_agent.py# Entry point to TRAIN (environment wrapper, loop)
├── saved_models/     # Saved model weights (.h5)
├── data/             # Logs and other generated data
├── requirements.txt  # Dependencies
├── .gitignore
└── README.md         # This file
```


