from RL.player import AlphaZeroPlayer
from RL.PVNet import PolicyValueNet
from RL.board import Board
import random

class rl(object):

    def __init__(self, args):
        best_policy = PolicyValueNet(
            load_path=args.load_path, 
            device=args.device, 
            internal_model=args.internal_model, 
            k=args.k)
        self.alphazero_player = AlphaZeroPlayer(
            best_policy.policy_value_fn, 
            c_puct=5, n_play=args.n_play, 
            k=args.k)

    def reset(self, chessboard, robot_color, last_drop):
        self.board = Board(robot_color, chessboard, last_drop)

    def get_best_drop(self):
        if len(self.board.drops) == 0:
            return (7, 7)
        if len(self.board.drops) == 1:
            low, high = 6, 8
            first_drop = list(self.board.drops.keys())[0]
            second_drops = []
            for i in range(low, high + 1):
                for j in range(low, high + 1):
                    if first_drop[0] != i or first_drop[1] != j:
                        second_drops.append([i, j])
            return second_drops[random.randint(0, (high - low + 1) ** 2 - 2)]

        move, _ = self.alphazero_player.get_move(self.board)
        print(move)
        return move
        
def get_drop(args, chessboard, robot_color, last_drop, rl_robot):
    """Get a drop from ai-RL"""
    rl_robot.reset(chessboard, robot_color, last_drop)
    return rl_robot.get_best_drop()