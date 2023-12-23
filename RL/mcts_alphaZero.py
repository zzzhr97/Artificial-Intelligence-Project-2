import numpy as np
import copy

dirichlet_alpha = 0.1

BLACK = -1
WHITE = 1
total_players = {BLACK: "BLACK", WHITE: "WHITE"}

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

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

        if len(self._children) == 0:
            print(len(self._children), action_priors)
            input()

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
    """An implementation of MCTS alphaZero"""

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

        # evaluate current board
        # return a list of (action, probability) tuples 
        # and a value in [-1, 1]
        action_probs, leaf_value = self._policy(board)

        end, winner = board.game_end()
        if self._root._children == {} and end:
            print(self._root._children, winner, board.states)
            raise Exception("ERROR: the root node has no children")

        if not end:
            node.expand(action_probs)
        else:

            # game end, return 0.0 for tie
            # 1.0 for current player win 
            # -1.0 for current player lose
            if winner == 0:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == board.get_current_player() else -1.0

        # update value recursively
        node.update_recursive(-leaf_value)

    def get_move_probs(self, board, temp=1e-3):
        """
        Simulate self._n_playout times from the given board,
        and return the actions and their probabilities.
        Param:
        - board: the current game board.
        - temp: temperature parameter in (0, 1] controls the level of exploration.

        Return: 
        - acts: a list of actions, like ((1,2), (5,5), ...).
        - act_probs: corresponding probabilities based on visit counts,
            like (10, 14, ...).
        """
        for n in range(self._n_playout):
            board_copy = copy.deepcopy(board)
            self._playout(board_copy)

        # calculate move probabilities based on visit counts
        act_visits = [(act, node._n_visits)
            for act, node in self._root._children.items()]
        
        acts, visits = zip(*act_visits)

        # temp：温度参数，控制探索的程度
        # 在高温度下，概率分布更加均匀，增加了对非最优动作的探索；
        # 在低温度下，概率分布更加尖锐，更倾向于选择 visit 次数较多的动作。
        # T -> inf，所有动作的概率都接近相等；
        # T -> 0，则接近与max操作，只有最大概率的动作的概率为1，其余为0。

        # use temperature parameter for exploration
        # calculate the probabilities of each action based on visit counts
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

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

    def __init__(self, 
            policy_value_function,
            c_puct=5, 
            n_playout=2000, 
            is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move((-1, -1))

    def get_action(self, board, temp=1e-3, return_prob=0):
        """
        Get the action based on MCTS.
        Param:
        - board: the current game board.
        - temp: temperature parameter in (0, 1] controls the level of exploration.
        - return_prob: whether to return the move probabilities

        Return:
        - move: the selected action.
        - move_probs: the move probabilities of each action(15*15 numpy array).
        """
        available_moves = board.availables  
        move_probs = np.zeros([board.width, board.height])

        if len(available_moves) > 0:

            # get all the probabilities of the children nodes
            acts, probs = self.mcts.get_move_probs(board, temp)
            acts, probs = np.array(acts), np.array(probs)

            if len(acts) > 0:   
                move_probs[acts[:, 0], acts[:, 1]] = probs

            # self play
            if self._is_selfplay:
                # add Dirichlet Noise for exploration 
                move_idx = np.random.choice(
                    len(acts),
                    p=0.75*probs + 0.25*np.random.dirichlet(dirichlet_alpha * np.ones(len(probs)))
                )
                move = tuple(acts[move_idx]) # like (2,3)

                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)

            # not self play, select by probs
            else:
                # temp = 1e-3: almost the same as choosing the action with the highest prob
                move_idx = np.random.choice(len(acts), p=probs)
                move = tuple(acts[move_idx])

                # reset the root node
                self.mcts.update_with_move((-1, -1))
                print(f"AI move: {move[0]}, {move[1]}")

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return f"MCTS {total_players[self.player]}"
