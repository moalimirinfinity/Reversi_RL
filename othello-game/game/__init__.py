"""
Othello game package - contains all game logic and components
"""

from .constants import *
from .board import Board
from .game import OthelloGame

__all__ = ['Board', 'OthelloGame', 'BOARD_SIZE', 'CELL_SIZE', 'MARGIN',
           'WINDOW_WIDTH', 'WINDOW_HEIGHT', 'FPS', 'BLACK', 'WHITE',
           'GREEN', 'BROWN', 'HIGHLIGHT', 'RUNNING', 'GAME_OVER']