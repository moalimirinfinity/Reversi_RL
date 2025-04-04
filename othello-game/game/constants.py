# Game constants
BOARD_SIZE = 8
CELL_SIZE = 80
MARGIN = 50
WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE + 2 * MARGIN
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * MARGIN + 60  # Extra space for UI
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (34, 139, 34)  # Green
DARK_WOOD = (101, 67, 33)    # Wooden frame
HIGHLIGHT = (255, 255, 0)    # Yellow
SHADOW = (50, 50, 50)        # Piece shadow
HIGHLIGHT_SHADOW = (200, 200, 0)  # Highlight shadow

# Piece colors
WHITE_PIECE = (240, 240, 240)
BLACK_PIECE = (30, 30, 30)
WHITE_HIGHLIGHT = (255, 255, 255)
BLACK_HIGHLIGHT = (80, 80, 80)

# Game states
RUNNING = 0
GAME_OVER = 1