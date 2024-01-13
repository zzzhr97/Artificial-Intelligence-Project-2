# DEL
from board import Board, Gomoku
# from mcts_pure import MCTSPlayer as MCTS_Pure
# from mcts_alphaZero import MCTSPlayer
from player import RandomPlayer, AlphaZeroPlayer
from PVNet import PolicyValueNet

# 0 for pure, 1 for alphazero
player = 1

BLACK = -1
WHITE = 1
total_players = {BLACK: "BLACK", WHITE: "WHITE"}
load_path = './RL/models/policy_Simple__best_epoch8350.pth'
internal_model = 'Simple'

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_move(self, board):
        try:
            location = input("Your move:")
            if isinstance(location, str): 
                location = tuple(int(n, 10) for n in location.split(","))
            move = location
        except Exception as e:
            move = (-1, -1)
        if move == (-1, -1) or move not in board.availables:
            print("invalid move")
            move = self.get_move(board)
        return move, None

    def __str__(self):
        return f"Human {total_players[self.player]}"


def run():
    n = 5
    try:
        board = Board()
        game = Gomoku(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy

        best_policy = PolicyValueNet(load_path=load_path, device='cuda', internal_model=internal_model)
        mcts_player = AlphaZeroPlayer(best_policy.policy_value_fn, c_puct=5, n_play=1000) 

        if player == 0:
            mcts_player = RandomPlayer(c_puct=5, n_play=2000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.play_game(human, mcts_player, start_player=WHITE, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()