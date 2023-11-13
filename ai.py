import numpy as np
import random

# max: black / -1, min: white / 1
INF_VALUE = {-1: - 2 ** 16, 1: 2 ** 16 - 1}

class State(object):

    def __init__(self, board, color, last_drop, depth, shuffle=False):
        self.board = np.array(board)
        self.color = color          # this drop is by <color>
        self.last_drop = last_drop  # last drop is by <-color>
        self.depth = depth
        self.shuffle = shuffle

    def legal_drops(self):
        """Get all the legal drops of the current state"""
        drops = np.column_stack(np.where(self.board == 0))
        if self.shuffle:
            idx = np.random.permutation(drops.shape[0])
            drops = drops[idx]
        return drops

    def next(self, drop):
        """Return a new state after a given drop: [row, column]"""
        new_board = np.array(self.board)
        new_board[drop[0], drop[1]] = self.color
        new_state = State(new_board,
                        - self.color,
                        drop,
                        self.depth - 1,
                        self.shuffle,
        )
        return new_state

class ai(object):
    def __init__(self):
        pass

    def get_first_drop(self):
        low = 6
        high = 8
        first_drops = []
        for i in range(low, high + 1):
            for j in range(low, high + 1):
                first_drops.append([i, j])
        return first_drops[random.randint(0, (high - low + 1) ** 2 - 1)]

    def is_win(self, board, pos):
        if self.__horizontal(board, pos):
            return True
        if self.__vertical(board, pos):
            return True
        if self.__main_diag(board, pos):
            return True
        if self.__cont_diag(board, pos):
            return True
        return False

    def is_full(self, board):
        return np.sum(np.abs(board)) == 225

    def __horizontal(self, board, pos):
        r, c = pos
        left = c-4 if c-4 > 0 else 0
        right = c+4 if c+4 < 15 else 14
        check = []
        for i in range(left, right-3):
            check.append(int(abs(sum(board[r, i:i+5])) == 5))
        return sum(check) == 1

    def __vertical(self, board, pos):
        r, c = pos
        top = r-4 if r-4 > 0 else 0
        bottom = r+4 if r+4 < 15 else 14
        check = []
        for i in range(top, bottom-3):
            check.append(int(abs(sum(board[i:i+5, c])) == 5))
        return sum(check) == 1

    def __main_diag(self, board, pos):
        r, c = pos
        left, top, right, bottom = 0, 0, 0, 0
        if r >= c:
            left = c-4 if c-4 > 0 else 0
            bottom = r+4 if r+4 < 15 else 14
            right = bottom - (r-c)
            top = left + r-c
        else:
            right = c+4 if c+4 < 15 else 14
            top = r-4 if r-4 > 0 else 0
            left = top+c-r
            bottom = right-(c-r)

        check = []
        if right-left > 3:
            for i in range(right-left-3):
                col = np.arange(left+i, left+i+5)
                row = np.arange(top+i, top+i+5)
                check.append(int(abs(sum(board[row, col])) == 5))
        return sum(check) == 1

    def __cont_diag(self, board, pos):
        r, c = pos
        left, top, right, bottom = 0, 0, 0, 0
        if r + c <= 14:
            top = r-4 if r-4 > 0 else 0
            left = c-4 if c-4 > 0 else 0
            bottom = r+c-left
            right = r+c-top
        else:
            bottom = r+4 if r+4 < 15 else 14
            right = c+4 if c+4 < 15 else 14
            top = r+c-right
            left = r+c-bottom

        check = []
        if right-left > 3:
            for i in range(right-left-3):
                col = np.arange(left+i, left+i+5)
                row = np.arange(bottom-i, bottom-i-5, -1)
                check.append(int(abs(sum(board[row, col])) == 5))
        return sum(check) == 1