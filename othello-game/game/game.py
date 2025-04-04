import pygame
from .constants import *
from .board import Board


class OthelloGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Othello')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)

        self.board = Board()
        self.current_player = 1  # White starts
        self.valid_moves = self.board.get_valid_moves(self.current_player)
        self.state = RUNNING
        self.highlighted = None

        # Load graphics (you'll need to create these assets)
        self.board_img = pygame.image.load('assets/images/board.png').convert()
        self.board_img = pygame.transform.scale(self.board_img,
                                                (BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE))
        self.black_piece = pygame.image.load('assets/images/black_piece.png').convert_alpha()
        self.black_piece = pygame.transform.scale(self.black_piece, (CELL_SIZE - 10, CELL_SIZE - 10))
        self.white_piece = pygame.image.load('assets/images/white_piece.png').convert_alpha()
        self.white_piece = pygame.transform.scale(self.white_piece, (CELL_SIZE - 10, CELL_SIZE - 10))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            self._draw()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    def _handle_mouse_motion(self, pos):
        if self.state == GAME_OVER:
            return

        row, col = self._get_cell_from_pos(pos)
        if (row, col) in self.valid_moves:
            self.highlighted = (row, col)
        else:
            self.highlighted = None

    def _handle_click(self, pos):
        if self.state == GAME_OVER:
            return

        row, col = self._get_cell_from_pos(pos)
        if (row, col) in self.valid_moves:
            self.board.make_move(row, col, self.current_player)
            self._switch_player()

    def _switch_player(self):
        self.current_player *= -1
        self.valid_moves = self.board.get_valid_moves(self.current_player)

        if not self.valid_moves:
            # Current player has no moves, check if game is over
            self.current_player *= -1
            self.valid_moves = self.board.get_valid_moves(self.current_player)

            if not self.valid_moves:
                self.state = GAME_OVER

    def _get_cell_from_pos(self, pos):
        x, y = pos
        row = (y - MARGIN) // CELL_SIZE
        col = (x - MARGIN) // CELL_SIZE
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return row, col
        return None, None

    def _draw(self):
        # Draw board background
        self.screen.fill(BROWN)

        # Draw board
        board_rect = pygame.Rect(MARGIN, MARGIN,
                                 BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE)
        self.screen.blit(self.board_img, board_rect)

        # Draw grid lines
        for i in range(BOARD_SIZE + 1):
            # Horizontal lines
            pygame.draw.line(self.screen, BLACK,
                             (MARGIN, MARGIN + i * CELL_SIZE),
                             (MARGIN + BOARD_SIZE * CELL_SIZE, MARGIN + i * CELL_SIZE), 2)
            # Vertical lines
            pygame.draw.line(self.screen, BLACK,
                             (MARGIN + i * CELL_SIZE, MARGIN),
                             (MARGIN + i * CELL_SIZE, MARGIN + BOARD_SIZE * CELL_SIZE), 2)

        # Draw pieces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board.grid[row][col] == 1:  # White
                    piece_pos = (MARGIN + col * CELL_SIZE + CELL_SIZE // 2,
                                 MARGIN + row * CELL_SIZE + CELL_SIZE // 2)
                    self.screen.blit(self.white_piece,
                                     (piece_pos[0] - self.white_piece.get_width() // 2,
                                      piece_pos[1] - self.white_piece.get_height() // 2))
                elif self.board.grid[row][col] == -1:  # Black
                    piece_pos = (MARGIN + col * CELL_SIZE + CELL_SIZE // 2,
                                 MARGIN + row * CELL_SIZE + CELL_SIZE // 2)
                    self.screen.blit(self.black_piece,
                                     (piece_pos[0] - self.black_piece.get_width() // 2,
                                      piece_pos[1] - self.black_piece.get_height() // 2))

        # Draw highlighted valid move
        if self.highlighted:
            row, col = self.highlighted
            highlight_rect = pygame.Rect(MARGIN + col * CELL_SIZE + 2,
                                         MARGIN + row * CELL_SIZE + 2,
                                         CELL_SIZE - 4, CELL_SIZE - 4)
            pygame.draw.rect(self.screen, HIGHLIGHT, highlight_rect, 3)

        # Draw score
        white_score, black_score = self.board.get_score()
        score_text = f"White: {white_score}  Black: {black_score}"
        score_surface = self.font.render(score_text, True, WHITE)
        self.screen.blit(score_surface, (MARGIN, 10))

        # Draw current player indicator
        player_text = f"Current: {'White' if self.current_player == 1 else 'Black'}"
        player_surface = self.font.render(player_text, True, WHITE)
        self.screen.blit(player_surface, (WINDOW_WIDTH - MARGIN - player_surface.get_width(), 10))

        # Draw game over message
        if self.state == GAME_OVER:
            winner = self._get_winner()
            if winner == 0:
                message = "Game Over! It's a draw!"
            else:
                message = f"Game Over! {'White' if winner == 1 else 'Black'} wins!"

            msg_surface = self.font.render(message, True, WHITE)
            msg_rect = msg_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 20))
            pygame.draw.rect(self.screen, BLACK,
                             (msg_rect.x - 10, msg_rect.y - 10,
                              msg_rect.width + 20, msg_rect.height + 20))
            self.screen.blit(msg_surface, msg_rect)

    def _get_winner(self):
        white, black = self.board.get_score()
        if white > black:
            return 1
        elif black > white:
            return -1
        else:
            return 0