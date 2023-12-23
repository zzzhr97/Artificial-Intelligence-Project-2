import minimax.minimax as minimax
import RL.rl as rl
import numpy as np

def robot(args, chessboard, robot_color, last_drop):
    # chessboard    a 15*15 ndarray, 0 is empty, -1 is black, 1 is white
    # robot_color   a interger, -1 is black, 1 is white
    # last_drop     a tuple (r,c), r is row of chessboard, c is column of chessboard

    # RETURN:       a tuple (r,c), which is location of robot to drop piece in this turn

    if args.ai == 'minimax':
        r, c = minimax.get_drop(args, chessboard, robot_color, last_drop)
    elif args.ai == 'rl':
        r, c = rl.get_drop(args, chessboard, robot_color, last_drop)
    else:
        r, c = 0, 0

    return (r, c)

