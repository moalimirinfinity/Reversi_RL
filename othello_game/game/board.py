import numpy as np
from .constants import *


class Board:
    def __init__(self):
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        # Initial setup
        self.grid[3][3] = self.grid[4][4] = 1  # White
        self.grid[3][4] = self.grid[4][3] = -1  # Black

    def is_valid_move(self, row, col, player):
        if self.grid[row][col] != 0:
            return False

        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
            if self._check_direction(row, col, dr, dc, player):
                return True
        return False

    def _check_direction(self, row, col, dr, dc, player):
        r, c = row + dr, col + dc
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return False
        if self.grid[r][c] != -player:
            return False

        r += dr
        c += dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if self.grid[r][c] == 0:
                return False
            if self.grid[r][c] == player:
                return True
            r += dr
            c += dc
        return False

    def make_move(self, row, col, player):
        if not self.is_valid_move(row, col, player):
            return False

        self.grid[row][col] = player
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
            self._flip_in_direction(row, col, dr, dc, player)
        return True

    def _flip_in_direction(self, row, col, dr, dc, player):
        if not self._check_direction(row, col, dr, dc, player):
            return

        r, c = row + dr, col + dc
        while self.grid[r][c] == -player:
            self.grid[r][c] = player
            r += dr
            c += dc

    def get_valid_moves(self, player):
        moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.is_valid_move(row, col, player):
                    moves.append((row, col))
        return moves

    def get_score(self):
        white = np.sum(self.grid == 1)
        black = np.sum(self.grid == -1)
        return white, black