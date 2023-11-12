import numpy as np
import tqdm
from ai import ai
from eval import evaluate_func

# max: black / -1, min: white / 1
INF_VALUE = {-1: - 2 ** 16, 1: 2 ** 16 - 1}

class State(object):

    def __init__(self, board, color, last_drop, depth):
        self.board = np.array(board)
        self.color = color          # this drop is by <color>
        self.last_drop = last_drop  # last drop is by <-color>
        self.depth = depth

    def legal_drop(self):
        """Get all the legal drops of the current state"""
        return np.column_stack(np.where(self.board == 0))

    def next(self, drop):
        """Return a new state after a given drop: [row, column]"""
        new_board = np.array(self.board)
        new_board[drop[0], drop[1]] = self.color
        new_state = State(new_board,
                        - self.color,
                        drop,
                        self.depth - 1,
        )
        return new_state

class minimax(ai):

    def __init__(self, args, chessboard, robot_color, last_drop):
        super(minimax, self).__init__()
        self.init_state = State(chessboard, robot_color, last_drop, args.depth)

    def get_best_drop(self):
        """Get the best drop in given init_state"""
        if np.sum(abs(self.init_state.board)) == 0:
            return self.get_first_drop()

        # best_drop = [row, colomn, value]
        best_drop = self._minimax(self.init_state, INF_VALUE[-1], INF_VALUE[1])
        return best_drop[:2]
    
    def _minimax(self, state, a, b):
        """Get the next best drop-value: [row, column, value]"""
        # the last drop leads to winning
        if self.is_win(state.board, state.last_drop):
            # debug
            #print(f"win! color: {state.color}")
            return [*state.last_drop, INF_VALUE[state.color]]
        
        if state.depth == 0:
            # debug
            #print("depth stop!")
            return [*state.last_drop, evaluate_func(state)]
        
        # next drop-value: [row, column, value]
        next_dv = [-1, -1, INF_VALUE[state.color]]
        
        if state.color == -1:   # black
            for drop in state.legal_drop():

                new_state = state.next(drop)
                # debug:
                #print(f"max drop: {drop}")
                tmp_dv = self._minimax(new_state, a, b)
                # debug:
                #print(f"max dv: {tmp_dv}")
                if tmp_dv[2] >= next_dv[2]:
                    next_dv = tmp_dv
                if next_dv[2] >= b:
                    return next_dv
                a = max(a, next_dv[2])

        else:   # white
            for drop in state.legal_drop():

                new_state = state.next(drop)
                # debug:
                #print(f"min drop: {drop}")
                tmp_dv = self._minimax(new_state, a, b)
                # debug:
                #print(f"min dv: {tmp_dv}")
                if tmp_dv[2] <= next_dv[2]:
                    next_dv = tmp_dv
                if next_dv[2] <= a:
                    return next_dv
                b = min(b, next_dv[2])

        assert next_dv[0] != -1, next_dv

        return next_dv


def get_drop(args, chessboard, robot_color, last_drop):
    """Get a drop from ai-minimax"""
    robot = minimax(args, chessboard, robot_color, last_drop)
    return robot.get_best_drop()