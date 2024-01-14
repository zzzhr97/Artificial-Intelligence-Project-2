import numpy as np
import copy
from tqdm import tqdm

class TreeNode(object):
    """
    A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prob):
        # parent node
        self.parent = parent

        # child nodes
        # e.g. {(0,3): TreeNode1, (5,5): TreeNode2, ...}
        self.children = {} 

        # number of visits
        self.N = 0

        # Q value
        self.Q = 0

        # U value
        self.U = 0

        # probabilit
        self.P = prob

    def expand(self, move_probs):
        """
        Expand tree.

        Param:
        - move_probs: [((1,2), 0.2), ...]
        """
        for move, prob in move_probs:
            move = tuple(move)
            if move not in self.children:
                self.children[move] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        Select move by UCB.

        Return: 
        - A tuple of (move, next_node)
        """
        return max(self.children.items(), key=lambda move_node: move_node[1].get_UCB(c_puct))

    def update(self, leaf_value):
        """
        Update node values from leaf evaluation.

        Param:
        - leaf_value: the value of subtree evaluation from the current 
            player's perspective.
        """
        self.N += 1
        self.Q += 1.0 * (leaf_value - self.Q) / self.N

    def update_recursive(self, leaf_value):
        """
        Apply update recursively for all ancestors.
        """
        # Update parent first
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_UCB(self, c_puct):
        """
        Param:
        - c_puct: a number (>0) controlling the weight of Q and U in UCB.

        Return:
        - UCB value of this node.
        """
        self.U = (c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N))
        return self.Q + self.U

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None
    
class MCTS(object):
    """Monte Carlo Tree."""
    def __init__(self, policy_value_fn, c_puct=5, n_play=10000):
        """
        Param:
        - policy_value_fn: a function that takes in a board and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        - c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        - n_play: number of simulations to run for each move.
        """
        self.root = TreeNode(None, 1.0)
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_play = n_play

    def simulate(self, board):
        """Simulate n_play times."""
        with tqdm(total=self.n_play, desc="Thinking...") as pbar:
            for i in range(self.n_play):
                board_copy = copy.deepcopy(board)
                self.playout(board_copy)
                pbar.update(1)
            pbar.set_description("Done...")

    def playout(self, board):
        """
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        Param:
        - board: a game board
        """
        node = self.root
        while(1):
            if node.is_leaf():
                break

            # select next best move
            action, node = node.select(self.c_puct)

            # do the move in the board
            board.update_board(action)

        # evaluate current board
        # return a list of (action, probability) tuples 
        # and a value in [-1, 1]
        move_probs, leaf_value = self.policy_value_fn(board)

        end, winner = board.is_end()
        if self.root.children == {} and end:
            print(self.root.children, winner, board.drops)
            raise Exception("ERROR: the root node has no children")

        if not end:
            node.expand(move_probs)

        # get true leaf value
        tmp_value = self.get_leaf_value(board, end, winner)
        if tmp_value:
            leaf_value = tmp_value

        # update value recursively
        node.update_recursive(-leaf_value)

    def get_leaf_value(self, board, end, winner):
        raise NotImplementedError
    
    def get_end_value(self, winner, cur_player):
        """
        Get value while game is over.
        - If tie, return 0.0
        - If current player wins, return 1.0
        - If opponent player wins, return -1.0
        """
        if winner == 0:  
            return 0.0
        else:
            return 1.0 if winner == cur_player else -1.0

    def move_root(self, new_move):
        """Move root due to a new move."""
        assert new_move in self.root.children
        self.root = self.root.children[new_move]
        self.root.parent = None

    def reset_root(self):
        """Reset root and tree."""
        self.root = TreeNode(None, 1.0)

    def __str__(self):
        raise NotImplementedError

class AlphaZeroTree(MCTS):
    """An implementation of alphaZero tree."""
    def __init__(self, policy_value_fn, c_puct=5, n_play=10000, k=0):
        super().__init__(policy_value_fn, c_puct, n_play)
        self.k = k

    def get_leaf_value(self, board, end, winner):
        """
        Given the game result, get the leaf value.
        - If not end, return None.
        - If end, get end value.
        """
        if not end:
            return None
        
        cur_player = board.get_cur_player()
        return self.get_end_value(winner, cur_player)
    
    # temperature：温度参数，控制探索的程度
    # 在高温度下，概率分布更加均匀，增加了对非最优动作的探索；
    # 在低温度下，概率分布更加尖锐，更倾向于选择 visit 次数较多的动作。
    # T -> inf，所有动作的概率都接近相等；
    # T -> 0，则接近与max操作，只有最大概率的动作的概率为1，其余为0。
    def get_pi(self, board, temperature=1e-3):
        """
        Simulate self.n_play times from the given board,
        and return the moves and their probabilities.
        Param:
        - board: the current game board.
        - temperature: temperature parameter in (0, 1] controls the level of exploration.

        Return: 
        - moves: a list of moves, like ((1,2), (5,5), ...).
        - move_probs: corresponding probabilities based on visit counts,
            like (10, 14, ...).
        """
        self.simulate(board)
        move_visits = [(move, node.N) for move, node in self.root.children.items()]
        moves, visits = self.mask(board, move_visits)

        print(moves, visits)

        # moves, visits = zip(*move_visits)
        move_probs = self.softmax(1.0/temperature * np.log(np.array(visits) + 1e-10))
        return moves, move_probs
    
    def mask(self, board, move_probs):
        mask_matrix = np.full((15, 15), self.k)
        for drop in board.drops:
            self._set_loc(mask_matrix, drop[0]+1, drop[1])
            self._set_loc(mask_matrix, drop[0]-1, drop[1])
            self._set_loc(mask_matrix, drop[0], drop[1]+1)
            self._set_loc(mask_matrix, drop[0], drop[1]-1)
            self._set_loc(mask_matrix, drop[0]+1, drop[1]+1)
            self._set_loc(mask_matrix, drop[0]-1, drop[1]-1)
            self._set_loc(mask_matrix, drop[0]+1, drop[1]-1)
            self._set_loc(mask_matrix, drop[0]-1, drop[1]+1)
        
        new_move_probs = []
        for move_prob in move_probs:
            new_move_prob = (move_prob[0], move_prob[1] * mask_matrix[move_prob[0][0], move_prob[0][1]])
            new_move_probs.append(new_move_prob)
        return zip(*new_move_probs)
    
    def _set_loc(self, mask_matrix, x, y):
        """Set the adjacent locations of the given location"""
        if 0 <= x < 15 and 0 <= y < 15:
            mask_matrix[x, y] = 1
    
    def softmax(self, x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

    def __str__(self):
        return "AlphaZeroTree"
    
class RandomTree(MCTS):
    """An implementation of random tree."""
    def __init__(self, policy_value_fn, c_puct=5, n_play=10000):
        super().__init__(policy_value_fn, c_puct, n_play)

    def get_leaf_value(self, board, end, winner):
        """Get the value by random simulation to end."""
        for i in range(225):
            end, winner = board.is_end()
            if end:
                break

            move_probs = self.random_policy(board)

            # move once
            max_action = max(move_probs, key=lambda x: x[1])[0]
            board.update_board(max_action)

        assert end, "ERROR: the game is not end"
        cur_player = board.get_cur_player()
        return self.get_end_value(winner, cur_player)
    
    def random_policy(self, board):
        """Random policy function."""
        move_probs = np.random.rand(len(board.availables))
        return zip(board.availables, move_probs)
    
    def get_pi(self, board):
        """
        Simulate self.n_play times from the given board,
        and return the best action.
        Param:
        - board: the current game board

        Return: 
        - the best action
        """
        self.simulate(board)
        return max(self.root.children.items(), key=lambda node: node[1].N)[0]
    
    def __str__(self):
        return "RandomTree"