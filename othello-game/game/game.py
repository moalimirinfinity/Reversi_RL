import pygame
import numpy as np
from .constants import *
from .board import Board


class OthelloGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Othello - Programmatic Drawing')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.bold_font = pygame.font.SysFont('Arial', 32, bold=True)

        self.board = Board()
        self.current_player = 1  # White starts
        self.valid_moves = self.board.get_valid_moves(self.current_player)
        self.state = RUNNING
        self.highlighted = None

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
        # Draw wooden frame background
        self.screen.fill(DARK_WOOD)

        # Draw board background
        board_rect = pygame.Rect(
            MARGIN, MARGIN,
            BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE
        )
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect)

        # Draw grid lines with subtle shading
        for i in range(BOARD_SIZE + 1):
            # Horizontal lines
            pygame.draw.line(
                self.screen, BLACK,
                (MARGIN, MARGIN + i * CELL_SIZE),
                (MARGIN + BOARD_SIZE * CELL_SIZE, MARGIN + i * CELL_SIZE),
                2
            )
            # Vertical lines
            pygame.draw.line(
                self.screen, BLACK,
                (MARGIN + i * CELL_SIZE, MARGIN),
                (MARGIN + i * CELL_SIZE, MARGIN + BOARD_SIZE * CELL_SIZE),
                2
            )

        # Draw pieces with shadows for depth
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board.grid[row][col] != 0:
                    center_x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
                    center_y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
                    radius = CELL_SIZE // 2 - 5

                    # Draw shadow first
                    shadow_offset = 3
                    pygame.draw.circle(
                        self.screen, SHADOW,
                        (center_x + shadow_offset, center_y + shadow_offset),
                        radius
                    )

                    # Draw main piece
                    piece_color = WHITE_PIECE if self.board.grid[row][col] == 1 else BLACK_PIECE
                    pygame.draw.circle(
                        self.screen, piece_color,
                        (center_x, center_y), radius
                    )

                    # Add highlight to pieces
                    highlight_radius = radius - 10
                    if highlight_radius > 5:
                        highlight_color = WHITE_HIGHLIGHT if self.board.grid[row][col] == 1 else BLACK_HIGHLIGHT
                        pygame.draw.circle(
                            self.screen, highlight_color,
                            (center_x - radius // 3, center_y - radius // 3),
                            highlight_radius // 2
                        )

        # Draw highlighted valid move
        if self.highlighted:
            row, col = self.highlighted
            center_x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
            center_y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
            radius = CELL_SIZE // 3

            # Draw shadow for highlight
            pygame.draw.circle(
                self.screen, HIGHLIGHT_SHADOW,
                (center_x + 2, center_y + 2), radius
            )

            # Draw highlight circle
            pygame.draw.circle(
                self.screen, HIGHLIGHT,
                (center_x, center_y), radius
            )

        # Draw UI panel at bottom
        self._draw_ui()

    def _draw_ui(self):
        # Draw UI background
        ui_rect = pygame.Rect(
            0, WINDOW_HEIGHT - 60,
            WINDOW_WIDTH, 60
        )
        pygame.draw.rect(self.screen, DARK_WOOD, ui_rect)
        pygame.draw.line(
            self.screen, BLACK,
            (0, WINDOW_HEIGHT - 60),
            (WINDOW_WIDTH, WINDOW_HEIGHT - 60),
            2
        )

        # Draw score
        white_score, black_score = self.board.get_score()

        # White score
        white_text = f"White: {white_score}"
        white_surface = self.font.render(white_text, True, WHITE)
        white_pos = (MARGIN, WINDOW_HEIGHT - 50)
        pygame.draw.circle(
            self.screen, WHITE_PIECE,
            (white_pos[0] - 25, white_pos[1] + white_surface.get_height() // 2),
            10
        )
        self.screen.blit(white_surface, white_pos)

        # Black score
        black_text = f"Black: {black_score}"
        black_surface = self.font.render(black_text, True, WHITE)
        black_pos = (MARGIN + 150, WINDOW_HEIGHT - 50)
        pygame.draw.circle(
            self.screen, BLACK_PIECE,
            (black_pos[0] - 25, black_pos[1] + black_surface.get_height() // 2),
            10
        )
        self.screen.blit(black_surface, black_pos)

        # Current player indicator
        player_text = f"Current Turn: {'WHITE' if self.current_player == 1 else 'BLACK'}"
        player_surface = self.bold_font.render(player_text, True, WHITE)
        player_pos = (
            WINDOW_WIDTH - MARGIN - player_surface.get_width(),
            WINDOW_HEIGHT - 50
        )
        self.screen.blit(player_surface, player_pos)

        # Draw sample piece next to turn indicator
        sample_piece_pos = (
            WINDOW_WIDTH - MARGIN - player_surface.get_width() - 30,
            WINDOW_HEIGHT - 50 + player_surface.get_height() // 2
        )
        pygame.draw.circle(
            self.screen,
            WHITE_PIECE if self.current_player == 1 else BLACK_PIECE,
            sample_piece_pos, 12
        )

        # Game over message
        if self.state == GAME_OVER:
            self._draw_game_over_message()

    def _draw_game_over_message(self):
        winner = self._get_winner()
        if winner == 0:
            message = "Game Over! It's a draw!"
        else:
            winner_text = "WHITE" if winner == 1 else "BLACK"
            message = f"Game Over! {winner_text} wins!"

        # Create semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))

        # Draw message box
        msg_surface = self.bold_font.render(message, True, WHITE)
        msg_rect = msg_surface.get_rect(
            center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        )

        box_rect = pygame.Rect(
            msg_rect.x - 20, msg_rect.y - 20,
            msg_rect.width + 40, msg_rect.height + 40
        )
        pygame.draw.rect(self.screen, DARK_WOOD, box_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, box_rect, 2, border_radius=10)

        self.screen.blit(msg_surface, msg_rect)

        # Draw restart prompt
        restart_text = "Click anywhere to play again"
        restart_surface = self.font.render(restart_text, True, WHITE)
        restart_rect = restart_surface.get_rect(
            center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50)
        )
        self.screen.blit(restart_surface, restart_rect)

    def _get_winner(self):
        white, black = self.board.get_score()
        if white > black:
            return 1
        elif black > white:
            return -1
        else:
            return 0