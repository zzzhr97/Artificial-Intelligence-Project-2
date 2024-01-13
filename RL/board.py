import numpy as np
from tqdm import tqdm

BLACK = -1
WHITE = 1

class Board(object):
    """A 15 * 15 board for Gomoku game."""
    def __init__(self, start_player=BLACK, chessboard=None, last_move=(-1, -1)):
        # player BLACK moves first
        self.players = [BLACK, WHITE]  
        self.reset(start_player, chessboard, last_move)

    def reset(self, start_player=BLACK, chessboard=None, last_move=(-1, -1)):
        # start player
        self.cur_player = start_player

        # get available moves and drops
        self.convert_board(chessboard)

        # last move is moved by [-self.cur_player] !!
        self.last_move = last_move

    def convert_board(self, chessboard=None):
        """
        Convert 15*15 chessboard to available move set.
        - availables: 
            - [(x1, y1), (x2, y2), ...]
        - drops: details of each move.
            - key: move, like (3,4)
            - value: player number, like -1
        """
        self.availables = [(x,y) for x in range(15) for y in range(15)]
        self.drops = {}

        # print("chessboard: ", chessboard)

        if chessboard is not None:
            for i in range(15):
                for j in range(15):
                    if chessboard[i][j] != 0:
                        self.availables.remove((i, j))
                        self.drops[(i, j)] = chessboard[i][j]   

        # print("availables: ", self.availables)
        # print("drops: ", self.drops)

    def get_moves(self):
        """
        Get the existed moves of current and opposite player.

        Return:
            - cur_moves: moves of current player
            - opp_moves: moves of opposite player
        """
        moves, players = list(zip(*self.drops.items()))
        moves = np.array(moves)
        players = np.array(players)

        # extract moves of current and opposite player
        cur_moves = moves[players == self.cur_player]
        opp_moves = moves[players != self.cur_player]

        return cur_moves, opp_moves

    def get_state(self):
        """
        Get the board state of the current player.

        Return:
            - State: 4 * 15 * 15
        """

        cur_state = np.zeros((4, 15, 15))
        if self.drops:
            # get moves of current and opposite player
            cur_moves, opp_moves = self.get_moves()

            # current player moves
            if len(cur_moves) > 0:
                cur_state[0, cur_moves[:, 0], cur_moves[:, 1]] = 1.0
            
            # opposite player moves
            if len(opp_moves) > 0:
                cur_state[1, opp_moves[:, 0], opp_moves[:, 1]] = 1.0
            
            # last move location, occupying the last second plane
            cur_state[2][self.last_move] = 1.0
        
        # 1.0: turn of BLACK to move
        # 0.0: turn of WHITE to move
        if len(self.drops) % 2 == 0:
            cur_state[3][:, :] = 1.0 
        return cur_state
    
    def get_cur_player(self):
        return self.cur_player

    def update_board(self, move):
        self.drops[move] = self.cur_player # BLACK or WHITE
        self.availables.remove(move)

        # switch player
        self.cur_player = - self.cur_player
        self.last_move = move

    def is_end(self):
        """
        Check whether the game is ended.

        Return:
            - end: True or False
            - winner: 1 for BLACK, -1 for WHITE, 0 for tie
        
        Return examples:
            - True, 0: tie
            - True, 1: BLACK wins
        """
        win, winner = self.is_win()

        # end and win
        if win:
            return True, winner
        
        # end and tie
        elif self.is_full(self.drops):
            return True, 0
        
        # not end
        return False, 0
    
    def get_cur_board(self):
        cur_board = np.zeros((15, 15))

        # no move
        if len(self.drops) == 0:
            return cur_board
        
        # get moves of current and opposite player
        cur_moves, opp_moves = self.get_moves()
        if len(cur_moves) > 0:
            cur_board[cur_moves[:, 0], cur_moves[:, 1]] = self.cur_player
        if len(opp_moves) > 0:
            cur_board[opp_moves[:, 0], opp_moves[:, 1]] = -self.cur_player
        return cur_board
    
    def is_win(self):
        if len(self.drops) < 9:
            return False, 0
        
        cur_board = self.get_cur_board()
        
        if self.__horizontal(cur_board, self.last_move):
            return True, self.drops[self.last_move]
        if self.__vertical(cur_board, self.last_move):
            return True, self.drops[self.last_move]
        if self.__main_diag(cur_board, self.last_move):
            return True, self.drops[self.last_move]
        if self.__cont_diag(cur_board, self.last_move):
            return True, self.drops[self.last_move]
        return False, 0

    def is_full(self, drops):
        return len(drops) == 225

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


class Gomoku(object):

    def __init__(self, board):
        self.board = board
        self.reset()

    def reset(self, player=None, temperature=None, player1=None, player2=None, start_player=BLACK):
        self.board.reset(start_player)
        self.self_player = player 

        self.temperature = temperature
        self.states = []
        self.pi = []
        self.cur_players = []

        if player1 is not None and player2 is not None:
            self.cur_player = player1
            self.player1 = player1
            self.player2 = player2
            self.player1.set_player_ind(BLACK)
            self.player2.set_player_ind(WHITE)
            self.players = {BLACK: self.player1, WHITE: self.player2}

    def store_one_data(self, pi):
        """Store the current pi and state."""
        self.states.append(self.board.get_state())
        self.pi.append(pi)
        self.cur_players.append(self.board.get_cur_player())

    def self_move(self):
        """
        - Move one step and store data.
        - Self play.
        """
        # get move and pi 
        move, pi = self.self_player.get_move(self.board, temperature=self.temperature)
            
        # store the data
        self.store_one_data(pi)

        # move one step
        self.board.update_board(move)

    def player_move(self):
        """
        - Move one step.
        - Two players play. 
        """
        # get current player
        self.cur_player = self.players[self.board.get_cur_player()]

        # get move and pi
        move, pi = self.cur_player.get_move(self.board)
        assert len(move) == 2, f"Move error: {move}"

        # move one step
        self.board.update_board(move)

    def get_data(self, winner):
        """Given winner, get the training data."""
        # winner from the perspective of the current player of each state
        winners_curs = np.zeros(len(self.cur_players))

        # not tie
        if winner != 0:
            winners_curs[np.array(self.cur_players) == winner] = 1.0
            winners_curs[np.array(self.cur_players) != winner] = -1.0

        return winner, zip(self.states, self.pi, winners_curs)

    def self_play(self, player, temperature=1e-3):
        """ 
        Get self play data (state, MCTS_pr, winner_cur) for training.

        Return:
        - winner: ultimate winner
        - data: ((state, MCTS_pr, winner_cur), ...)
            - state: 4 * 15 * 15.
            - MCTS_pr: move probabilities from MCTS (15 * 15 numpy array).
            - winner_cur: winners from the perspective of the current player.
        """
        # index for moving
        idx = 0
        self.reset(player, temperature)

        # begin self play
        while True:
            # print for notation
            print(f"Move {idx}", end='\r')
            idx+=1

            # move one step and store data
            self.self_move()
            is_end, winner = self.board.is_end()
            if is_end:
                player.reset_player()
                return self.get_data(winner)

    def play_game(self, player1, player2, start_player=BLACK, is_shown=1):
        """
        Play Gomoku. Player1 for BLACK, player2 for WHITE.
        """
        assert start_player in (BLACK, WHITE), f"Player error: {start_player}"
        self.reset(player1=player1, player2=player2, start_player=start_player)
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            # move one step
            self.player_move()

            if is_shown:
                self.graphic(self.board, player1.player, player2.player)

            is_end, winner = self.board.is_end()
            if is_end:
                if is_shown:
                    if winner != 0:
                        print("Game end. Winner is", self.players[winner])
                    else:
                        print("Game end. Tie")
                return winner
            
    # delete
    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = 15
        height = 15

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(15):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(14, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(15):
                loc = (i, j)
                p = board.drops.get(loc, 0)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    