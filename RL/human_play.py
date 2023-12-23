# DEL
import pickle
from board import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from PVNet import PolicyValueNet

BLACK = -1
WHITE = 1
total_players = {BLACK: "BLACK", WHITE: "WHITE"}
load_path = 'best_policy.model'

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move:")
            if isinstance(location, str): 
                location = tuple(int(n, 10) for n in location.split(","))
            move = location
        except Exception as e:
            move = (-1, -1)
        if move == (-1, -1) or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return f"Human {total_players[self.player]}"


def run():
    n = 5
    width, height = 15, 15
    try:
        board = Board(width=width, height=height)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy

        # try:
        #     policy_param = pickle.load(open(load_path, 'rb'))
        # except:
        #     policy_param = pickle.load(open(load_path, 'rb'),
        #                             encoding='bytes')  # To support python3
        # best_policy = PolicyValueNet(width, height, policy_param)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=1000) 

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=WHITE, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()