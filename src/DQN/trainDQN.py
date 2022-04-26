
from src.DQN.LearnerBot import *
from src.DQN.utils import *

# import from the game repo

from src.player.RandomBot import *
from src.player.BuildAndRunBot import *
from src.Settings            import *
from src.interface.Color     import *
from src.interface.Board     import *
from src.interface.Pawn      import *
from src.action.PawnMove     import *
from src.action.FencePlacing import *
from src.Path                import *

from collections import deque
import numpy as np
import math


GRID_LENTH = 5
TOTAL_FENCES = 2

DefaultColorForPlayer = [
        Color.RED,
        Color.BLUE,
        Color.GREEN,
        Color.ORANGE
    ]

DefaultNameForPlayer = [
    "1",
    "2",
    "3",
    "4"
]

agent = LearnerBot(state_dim=52, action_dim=4)
player1 = agent
player2 = BuildAndRunBot()

def train(agent, max_t=1000,eps_start=1.0, eps_end = 0.01,
       eps_decay=0.996):
   """

   :param agent:
   :param max_t:
   :param eps_start:
   :param eps_end:
   :param eps_decay:
   :return:
   """

   eps = eps_start

   board = Board(5, 5, 32)
   playerCount = 2
   players = [agent, player2]

   for i in range(playerCount):
       if players[i].name is None:
           players[i].name = DefaultNameForPlayer[i]
       if players[i].color is None:
           players[i].color = DefaultColorForPlayer[i]

       # Initinialize player pawn
       players[i].pawn = Pawn(board, players[i])
       # Define player start positions and targets
       players[i].startPosition = board.startPosition(i)
       players[i].target = players[i].startPosition.row
       players[i].endPositions = board.endPositions(i)

   board.initStoredValidActions()
   state = board_to_state(board, players[0].name, TOTAL_FENCES)

   score = 0
   done = False
   for t in range(max_t):
       action = agent.play(state, eps)
       if isinstance(action, PawnMove):
           if math.abs(agent.target - action.fromCoord.row) > math.abs(agent.target - action.toCoord.row) : reward = 1
           elif math.abs(agent.target - action.fromCoord.row) < math.abs(agent.target - action.toCoord.row) : reward = -1
           else: reward = 0
           agent.movePawn(action.toCoord)
           # Check if the pawn has reach one of the player targets
           if agent.hasWon():
               done = True
       elif isinstance(action, FencePlacing):
           agent.placeFence(action.coord, action.direction)
           reward = 0
       next_state = board_to_state(board, players[0].name, TOTAL_FENCES)
       agent.step(state, action, reward, next_state, done)
       ## above step decides whether we will train(learn) the network
       ## actor (local_qnetwork) or we will fill the replay buffer
       ## if len replay buffer is equal to the batch size then we will
       ## train the network or otherwise we will add experience tuple in our
       ## replay buffer.
       state = next_state
       score += reward
       if done:
           break
       eps = max(eps * eps_decay, eps_end)  ## decrease the epsilon


    #torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

   return score

