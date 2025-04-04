import os
import datetime

class SimpleLogger:
    def __init__(self, log_dir, log_file="training_log.txt"):
        """
        Initializes a simple logger to print to console and save to a file.

        Args:
            log_dir (str): Directory where the log file will be saved.
            log_file (str): Name of the log file.
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_path = os.path.join(log_dir, log_file)
        self._log(f"--- Training Log Started: {datetime.datetime.now()} ---", init=True)

    def _log(self, message, init=False):
        """Internal method to print and write to file."""
        print(message)
        mode = 'w' if init else 'a'
        try:
            with open(self.log_path, mode) as f:
                f.write(message + '\n')
        except IOError as e:
            print(f"Warning: Could not write to log file {self.log_path}. Error: {e}")

    def log_episode(self, episode, total_steps, stats):
        """
        Logs summary statistics for a completed episode.

        Args:
            episode (int): The episode number.
            total_steps (int): Total training steps so far.
            stats (dict): A dictionary containing statistics like 'reward', 'steps', 'epsilon', 'result'.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = (
            f"[{timestamp}] Episode: {episode: <6} | "
            f"Total Steps: {total_steps: <8} | "
            f"Ep Steps: {stats.get('steps', 'N/A'): <4} | "
            f"Ep Reward: {stats.get('reward', 'N/A'): <6.2f} | "
            f"Result: {stats.get('result', 'N/A'): <10} | "
            f"Epsilon: {stats.get('epsilon', 'N/A'): <6.4f}"
        )
        self._log(log_message)

    def log_step(self, step, info):
        """
        Logs information at a specific step (less frequent).

        Args:
            step (int): The current global step number.
            info (dict): Dictionary containing step-specific info like 'epsilon'.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] Step: {step: <8} | Info: {info}"
        # Only print this less frequently to avoid console spam
        # self._log(log_message) # Uncomment if detailed step logging to file is needed
        print(log_message) # Print to console more often if desired

    def log_message(self, message):
        """Logs a general message."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log(f"[{timestamp}] INFO: {message}")