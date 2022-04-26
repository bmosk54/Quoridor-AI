#
#
#   This file contains functions that convert between the game
#   and the state, action from the RL engine.
#
#   Author: Xiaoyu Liu
#
#
#

from src.Settings            import *
from src.Path                import *
from src.Game                import *
from src.action.PawnMove     import *
from src.action.FencePlacing import *
from src.interface.Board     import *
from src.interface.Pawn      import *
from src.GridCoordniates     import *

import math
import torch


def recover_coords(grid_len, state, role):
    """
    extract coordinate from the state

    assume square grids, so that grid_len is a scalar, for a 5x5 grid, grid_len = 5

    return a GridCoordinates object representing the coordinate of the object

    """

    if role == 'player1':
        num = (state == 1).nonzero(as_tuple=False).item()

    elif role == 'player2':
        num = (state == 2).nonzero(as_tuple=False).item()

    x, y = math.ceil(num / grid_len - 1), num % grid_len - 1

    if y == -1: y += grid_len

    return GridCoordinates(x, y)


def board_to_state(board, player1_name, total_fences):
    """

    convert board to state

    state is a row vector, size = col * row + 2 (assume square board)

    """

    col, row = board.cols, board.rows
    state = torch.zeros([1, col * row + 2])

    # player 1

    x1, y1 = board.pawns[0].coord.row, board.pawns[0].coord.col

    state[:, x1 * row + y1] = 1

    # player 2

    x2, y2 = board.pawns[1].coord.row, board.pawns[1].coord.col

    state[:, x2 * row + y2] = 2

    # fences placed by player 1 and player 2

    fences_left_for_player1 = fences_left_for_player2 = total_fences/2

    for fence in board.fences:
        if fence.player == player1_name:
            state[:, x2 * row + y2 + row * col] = 3
            fences_left_for_player1 -= 1
        else:
            state[:, x2 * row + y2 + row * col] = 4
            fences_left_for_player2 -= 1

    state[:,-1] = fences_left_for_player2
    state[:, -2] = fences_left_for_player1

    return state


def move_to_action(move):
    """
    convert move to action

    """

    action = torch.zeros([1, 4])

    if isinstance(move, PawnMove):
        action[:, 1:] = -1
        if move.fromCoord.row - move.toCoord.row > 0:
            action[:, 0] = 1
        elif move.fromCoord.row - move.toCoord.row < 0:
            action[:, 0] = 2
        elif move.fromCoord.col - move.toCoord.col > 0:
            action[:, 0] = 3
        else:
            action[:, 0] = 4

    elif isinstance(move, FencePlacing):
        action[:, 0] = -1
        action[:, 1] = move.coord.row
        action[:, 2] = move.coord.col
        if move.direction == Fence.DIRECTION.VERTICAL:
            action[:, 3] = 0
        else:
            action[:, 3] = 1
    return action


def action_to_move(action, x, y):
    """
    convert action to move

    """
    if action[:, 0] == -1:
        if action[:, 3] == 0:
            move = FencePlacing(GridCoordinates(x, y), Fence.DIRECTION.VERTICAL)
        else:
            move = FencePlacing(GridCoordinates(x, y), Fence.DIRECTION.HORIZONTAL)
        return move

    elif action[:, 0] == 1:
        return PawnMove(GridCoordinates(x, y), GridCoordinates(x + 1, y))
    elif action[:, 0] == 2:
        return PawnMove(GridCoordinates(x, y), GridCoordinates(x - 1, y))
    elif action[:, 0] == 3:
        return PawnMove(GridCoordinates(x, y), GridCoordinates(x, y + 1))
    elif action[:, 0] == 4:
        return PawnMove(GridCoordinates(x, y), GridCoordinates(x, y - 1))
