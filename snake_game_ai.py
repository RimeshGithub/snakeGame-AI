import torch
import numpy as np
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet


class AI:
    def __init__(self):
        # Load trained model
        self.model = Linear_QNet(11, 256, 3)
        self.model.load_state_dict(torch.load('./model/model.pth'))
        self.model.eval()  # inference mode (VERY IMPORTANT)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)

    def get_action(self, state):
        final_move = [0, 0, 0]

        state0 = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():  # no gradients → faster
            prediction = self.model(state0)

        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move


def play():
    agent = AI()
    game = SnakeGameAI()

    while True:
        state = agent.get_state(game)
        final_move = agent.get_action(state)

        reward, done, score = game.play_step(final_move)

        if done:
            print("Game Over — Score:", score)
            game.reset()
            break


if __name__ == '__main__':
    play()
