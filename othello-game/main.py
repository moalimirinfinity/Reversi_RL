from game.game import OthelloGame


def main():
    while True:
        game = OthelloGame()
        game.run()

        # After game ends, ask if player wants to restart
        # (This is already handled in the game over screen clicks)


if __name__ == "__main__":
    main()