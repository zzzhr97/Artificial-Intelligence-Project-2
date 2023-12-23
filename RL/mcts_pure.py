import numpy as np
import copy

BLACK = -1
WHITE = 1
total_players = {BLACK: "BLACK", WHITE: "WHITE"}

def rollout_policy_fn(board):
    """Rollout policy function for the pure MCTS Player."""
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)

def policy_value_fn(board):
    """
    Param:
    - board: a board

    Return:
    - a list of (action, probability) tuples for each available move
        - example: [((1,2), 0.2), ...]
    - a score in [-1, 1] that gives the probability of the current player
    """
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """
    A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):

        # parent node
        self._parent = parent

        # child nodes
        # e.g. {(0,3): TreeNode1, (5,5): TreeNode2, ...}
        self._children = {} 

        # number of visits
        self._n_visits = 0

        # Q value
        self._Q = 0

        # U value
        self._U = 0

        # probability given by MCTS
        self._P = prior_p

    def expand(self, action_priors):
        """
        Expand tree by creating new children.

        Param:
        - action_priors: [((1,2), 0.2), ...]
        """
        for action, prob in action_priors:
            action = tuple(action)
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        Select action among children that gives maximum action value Q
        plus bonus u(P).

        Return: 
        - A tuple of (action, next_node)
        """
        return max(self._children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        Update node values from leaf evaluation.

        Param:
        - leaf_value: the value of subtree evaluation from the current 
            player's perspective.
        """
        # Count visit.
        self._n_visits += 1

        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        Apply update recursively for all ancestors.
        """
        # Update parent first
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        Param:
        - c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.

        Return:
        - UCB value for this node.
        """
        self._U = (c_puct * self._P *
            np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._U

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    """A simple implementation of MCTS"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        Param:
        - policy_value_fn: a function that takes in a board and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        - c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        - n_playout: number of simulations to run for each move.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, board):
        """
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        Param:
        - board: a game board
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break

            # select next best move
            action, node = node.select(self._c_puct)

            # do the move in the board
            board.do_move(action)

        # evaluate the leaf node by policy value function
        action_probs, _ = self._policy(board)

        end, winner = board.game_end()
        if not end:
            node.expand(action_probs)

        # evaluate the leaf node by random rollout to the end
        # 0: tie, 1: current player win, -1: opponent player win
        leaf_value = self._evaluate_rollout(board)

        # update value recursively
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, board, limit=1000):
        """
        Use the rollout policy to play until the end of the game.
        Param:
        - board: a game board
        - limit: the maximum number of moves to rollout

        Return:
        - the value of the current board from the perspective of the current
            +1: the current player wins
            -1: the opponent player wins
            0: tie
        """
        # current player
        player = board.get_current_player()

        for i in range(limit):
            end, winner = board.game_end()
            if end:
                break

            # use rollout policy to play
            action_probs = rollout_policy_fn(board)

            # select action by max probability
            max_action = max(action_probs, key=lambda x: x[1])[0]
            board.do_move(max_action)

        # for loop complete without break
        else:
            print("WARNING: rollout reaches move limit")

        # tie
        if winner == 0:  
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, board):
        """
        Simulate self._n_playout times from the given board,
        and return the best action.
        Param:
        - board: the current game board

        Return: 
        - the best action
        """
        for i in range(self._n_playout):
            board_copy = copy.deepcopy(board)
            self._playout(board_copy)

        # select the action with the highest visit count
        return max(self._root._children.items(),
            key=lambda node: node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """
        Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        # if last_move is in children of root
        # update the root node to that child
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None

        # otherwise, update the root node to a new TreeNode
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move((-1, -1))

    def get_action(self, board):
        available_moves = board.availables
        if len(available_moves) > 0:

            # simulate and get the best move
            move = self.mcts.get_move(board)

            # reset root node
            self.mcts.update_with_move((-1, -1))
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return f"MCTS {total_players[self.player]}"
