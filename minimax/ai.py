import numpy as np
import random 

# max: black / -1, min: white / 1
INF_VALUE = {-1: -9999999, 1: 9999999}

class State(object):

    def __init__(self, board, color, last_drop, depth, new_drops=None, locs=None, shuffle=False):
        self.board = np.array(board)
        self.color = color          
        # this drop is by <color>
        # last drop is by <-color>

        if new_drops is None:
            self.new_drops = []
        else:
            self.new_drops = list(new_drops)    # deep copy
        self.new_drops.append(last_drop)

        self.depth = depth
        self.shuffle = shuffle

        if locs is not None:
            drops = [last_drop]
            self.adjacent_locations = np.array(locs)
        else:
            drops = np.column_stack(np.where(self.board != 0))
            self.adjacent_locations = np.zeros_like(self.board)

        # update adjacent locations
        # drops = []
        # _set_loc = lambda x, y: \
        #     self.adjacent_locations.__setitem__((x, y), 1) \
        #     if 0 <= x < len(self.board) and 0 <= y < len(self.board) else None
        for [x, y] in drops:
            self._set_loc(x-1, y)
            self._set_loc(x, y-1)
            self._set_loc(x+1, y)
            self._set_loc(x, y+1)
            self._set_loc(x-1, y-1)
            self._set_loc(x+1, y+1)
            self._set_loc(x-1, y+1)
            self._set_loc(x+1, y-1)

    def _set_loc(self, x, y):
        """Set the adjacent locations of the given location"""
        if 0 <= x < len(self.board) and 0 <= y < len(self.board):
            self.adjacent_locations[x, y] = 1

    def legal_drops(self):
        """Get all the legal drops of the current state"""
        # return the adjacent drops
        assert self.adjacent_locations is not None, "adjacent_locations is None"
        #drops = np.column_stack(np.where(self.board == 0))
        drops = np.column_stack(np.where(np.logical_and(self.adjacent_locations == 1, self.board == 0)))
        if self.shuffle:
            idx = np.random.permutation(drops.shape[0])
            drops = drops[idx]
        return drops

    def next(self, drop):
        """Return a new state after a given drop: [row, column]"""
        new_board = np.array(self.board)
        new_board[drop[0], drop[1]] = self.color
        new_state = State(board=new_board,
                        color=-self.color,
                        last_drop=drop,
                        depth=self.depth-1,
                        new_drops=self.new_drops,
                        locs=self.adjacent_locations,
                        shuffle=self.shuffle,
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
    
    def get_second_drop(self, board):
        low = 6
        high = 8
        first_drop = np.column_stack(np.where(board != 0))[0]
        second_drops = []
        for i in range(low, high + 1):
            for j in range(low, high + 1):
                if first_drop[0] != i or first_drop[1] != j:
                    second_drops.append([i, j])
        return second_drops[random.randint(0, (high - low + 1) ** 2 - 2)]

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