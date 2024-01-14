import numpy as np
from mcts import AlphaZeroTree, RandomTree

BLACK = -1
WHITE = 1
total_players = {BLACK: "BLACK", WHITE: "WHITE"}

class player(object):
    """AI player."""
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.reset_root()

    def __str__(self):
        raise NotImplementedError

class AlphaZeroPlayer(player):
    """AI player based on AlphaZero."""
    def __init__(self, 
            policy_value_function,
            c_puct=5, 
            n_play=2000, 
            dirichlet_alpha=0.3,
            is_selfplay=0,
            k=0):
        super().__init__()
        self.mcts = AlphaZeroTree(policy_value_function, c_puct, n_play, k)
        self.is_selfplay = is_selfplay
        self.dirichlet_alpha = dirichlet_alpha
        self.k = k
        print("Alpha Zero Player: "
            f"[c_puct: {c_puct}] "
            f"[n_play: {n_play}] "
            f"[is_selfplay: {is_selfplay}] "
            f"[dirichlet_alpha: {self.dirichlet_alpha}] "
            f"[k: {k}] "
        )

    def get_move(self, board, temperature=1e-3):
        """
        Get the action based on AlphaZeroTree.
        Param:
        - board: the current game board.
        - temperature: temperature parameter in (0, 1] controls the level of exploration.

        Return:
        - move: the selected action.
        - pi: the move probabilities of each action(15*15 numpy array).
        """
        pi = np.zeros([15, 15])
        if len(board.availables) == 0:
            return None, None

        # get all the probabilities of the children nodes
        moves, probs = self.mcts.get_pi(board, temperature)
        moves, probs = np.array(moves), np.array(probs)

        if len(moves) > 0:   
            pi[moves[:, 0], moves[:, 1]] = probs

        # self play
        if self.is_selfplay:
            # add Dirichlet Noise for exploration 
            move_idx = np.random.choice(
                len(moves),
                p=0.75*probs + 0.25*np.random.dirichlet(self.dirichlet_alpha * np.ones(len(probs)))
            )
            move = tuple(moves[move_idx]) # like (2,3)

            # move root in the tree
            self.mcts.move_root(move)

        else:
            # temperature = 1e-3: almost the same as choosing the action with the highest prob
            move_idx = np.random.choice(len(moves), p=probs)
            move = tuple(moves[move_idx])

            # reset the root and get a new tree
            self.mcts.reset_root()

        return move, pi

    def __str__(self):
        return f"AlphaZero {total_players[self.player]}"
    

class RandomPlayer(player):
    """Random AI player."""
    def __init__(self, c_puct=5, n_play=2000):
        super().__init__()
        self.mcts = RandomTree(self.uniform_policy_value, c_puct, n_play)

    def get_move(self, board):
        if len(board.availables) == 0:
            return None, None
        move = self.mcts.get_pi(board)
        self.mcts.reset_root()
        return move, None
    
    def uniform_policy_value(self, board):
        """Get uniform distribution and zero value."""
        move_probs = np.ones(len(board.availables))/len(board.availables)
        return zip(board.availables, move_probs), 0

    def __str__(self):
        return f"Random {total_players[self.player]}"