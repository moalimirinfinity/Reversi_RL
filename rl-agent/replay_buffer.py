import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): Maximum size of buffer.
        """
        self.memory = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add(self, state, action_index, reward, next_state, done):
        """Add a new experience to memory."""
        experience = (state, action_index, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=min(batch_size, len(self.memory)))

        states = np.array([e[0] for e in experiences if e is not None])
        action_indices = np.array([e[1] for e in experiences if e is not None])
        rewards = np.array([e[2] for e in experiences if e is not None])
        next_states = np.array([e[3] for e in experiences if e is not None])
        dones = np.array([e[4] for e in experiences if e is not None]).astype(np.uint8)

        return (states, action_indices, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)