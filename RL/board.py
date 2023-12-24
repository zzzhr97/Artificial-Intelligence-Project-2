import numpy as np
from tqdm import tqdm

BLACK = -1
WHITE = 1

class Board(object):

    def __init__(self, width=15, height=15, n_win=5, start_player=BLACK):
        self.width = width
        self.height = height

        # number of pieces in a row to win
        self.n_win = n_win

        # player BLACK moves first
        self.players = [BLACK, WHITE]  

        # init board
        self.init_board(start_player)

    def init_board(self, start_player=BLACK):
        # start player
        self.current_player = start_player

        # all possible moves
        self.all_moves = [(x,y) for x in range(self.width) for y in range(self.height)]

        # available moves
        self.availables = list(self.all_moves)

        # last move is moved by [-self.current_player] !!
        self.last_move = (-1, -1)

        # states indicates the current board state
        # key: move, like (3,4)
        # value: player number, like -1
        self.states = {}

    def current_state(self):
        """
        return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        stack_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = list(zip(*self.states.items()))
            moves = np.array(moves)
            players = np.array(players)
            cur_moves = moves[players == self.current_player]
            opp_moves = moves[players != self.current_player]

            # current player moves
            if len(cur_moves) > 0:
                stack_state[0, cur_moves[:, 0], cur_moves[:, 1]] = 1.0
            
            # opposite player moves
            if len(opp_moves) > 0:
                stack_state[1, opp_moves[:, 0], opp_moves[:, 1]] = 1.0
            
            # last move location
            stack_state[2][self.last_move] = 1.0
        
        # 1.0: turn of BLACK to move
        # 0.0: turn of WHITE to move
        if len(self.states) % 2 == 0:
            stack_state[3][:, :] = 1.0 
        return stack_state

    def do_move(self, move):
        self.states[move] = self.current_player # BLACK or WHITE
        self.availables.remove(move)

        # switch player
        self.current_player = - self.current_player
        self.last_move = move

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()

        # win
        if win:
            return True, winner
        
        # tie
        elif self.is_full(self.states):
            return True, 0
        
        # not end
        return False, 0

    def get_current_player(self):
        return self.current_player
    
    def get_current_board(self):
        cur_board = np.zeros((self.width, self.height))

        # no move
        if len(self.states) == 0:
            return cur_board
        
        moves, players = list(zip(*self.states.items()))
        moves = np.array(moves)
        players = np.array(players)
        cur_moves = moves[players == self.current_player]
        opp_moves = moves[players != self.current_player]
        if len(cur_moves) > 0:
            cur_board[cur_moves[:, 0], cur_moves[:, 1]] = self.current_player
        if len(opp_moves) > 0:
            cur_board[opp_moves[:, 0], opp_moves[:, 1]] = -self.current_player
        return cur_board
    
    def has_a_winner(self):
        if len(self.states) < self.n_win * 2 - 1:
            return False, 0
        
        cur_board = self.get_current_board()
        
        if self.__horizontal(cur_board, self.last_move):
            return True, self.states[self.last_move]
        if self.__vertical(cur_board, self.last_move):
            return True, self.states[self.last_move]
        if self.__main_diag(cur_board, self.last_move):
            return True, self.states[self.last_move]
        if self.__cont_diag(cur_board, self.last_move):
            return True, self.states[self.last_move]
        return False, 0

    def is_full(self, states):
        return len(states) == 225

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


class Game(object):

    def __init__(self, board):
        self.board = board

    def start_self_play(self, player, temp=1e-3):
        """ 
        Start a self-play game and store data (state, MCTS_pr, winner_cur) for training.

        Return:
        - winner: ultimate winner
        - data: ((state, MCTS_pr, winner_cur), ...)
            - state: 4 * width * height.
            - MCTS_pr: move probabilities from MCTS (15 * 15 numpy array).
            - winner_cur: winners from the perspective of the current player.
        """
        states, mcts_prs, current_players = [], [], []
        self.board.init_board()

        idx = 0

        while True:

            print(f"Move {idx}", end='\r')
            idx+=1

            # get move and move probabilities from MCTS
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            
            # store the data
            states.append(self.board.current_state())
            mcts_prs.append(move_probs)
            current_players.append(self.board.current_player)

            # move one step
            self.board.do_move(move)

            # check if the game is ended
            end, winner = self.board.game_end()
            if end:

                # winner from the perspective of the current player of each state
                winners_curs = np.zeros(len(current_players))

                # not tie
                if winner != 0:
                    winners_curs[np.array(current_players) == winner] = 1.0
                    winners_curs[np.array(current_players) != winner] = -1.0
                    
                # reset MCTS root node
                player.reset_player()
                return winner, zip(states, mcts_prs, winners_curs)

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = (i, j)
                p = board.states.get(loc, 0)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=BLACK, is_shown=1):
        """
        Start a game between two players.
        player1 for BLACK, player2 for WHITE.
        """
        if start_player not in (BLACK, WHITE):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players # [BLACK, WHITE]
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)

            # move: like (3,4)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != 0:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    