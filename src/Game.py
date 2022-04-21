#
# Game.py
#
# @author    Alain Rinder
# @date      2017.06.02
# @version   0.1
#

from pdb import runeval
import random
#from player.BuildAndRunBot import BuildAndRunBot
from src.player.BuildAndRunBot import BuildAndRunBot
from src.player.MCTS_Bot import MCTS_Bot
from src.Settings            import *
from src.interface.Color     import *
from src.interface.Board     import *
from src.interface.Pawn      import *
#from src.player.Human        import *
#from src.player.BuildAndRunBot import *
from src.action.PawnMove     import *
from src.action.FencePlacing import *
from src.Path                import *
#from src.player.MCTS_Bot    import *
#from src.player.MCTS_Bot import *
import torch
import numpy as np
#from src.player.MCTS_Bot import MCTS_Bot as Mcts



class Game:
    """
    Define players and game parameters, and manage game rounds.
    """

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

    def __init__(self, players, cols = 9, rows = 9, totalFenceCount = 20, squareSize = 32, innerSize = None):
        if innerSize is None:
            innerSize = int(squareSize/8)
        self.totalFenceCount = totalFenceCount
        # Create board instance
        board = Board(self, cols, rows, squareSize, innerSize)
        # Support only 2 or 4 players
        playerCount = min(int(len(players)/2)*2, 4)
        self.players = []
        # For each player
        for i in range(playerCount):
            if not INTERFACE and isinstance(players[i], Human):
                raise Exception("Cannot launch a blind game with human players")
            # Define player name and color
            if players[i].name is None:
                players[i].name = Game.DefaultNameForPlayer[i]
            if players[i].color is None:
                players[i].color = Game.DefaultColorForPlayer[i]
            # Initinialize player pawn
            players[i].pawn = Pawn(board, players[i])
            # Define player start positions and targets
            players[i].startPosition = board.startPosition(i)
            players[i].endPositions = board.endPositions(i)
            self.players.append(players[i])
        self.board = board

    def start(self, roundCount = 1):
        """
        Launch a series of rounds; for each round, ask successively each player to play.
        """
        roundNumberZeroFill = len(str(roundCount))
        # For each round
        for roundNumber in range(1, roundCount + 1):
            # Reset board stored valid pawn moves & fence placings, and redraw empty grid
            self.board.initStoredValidActions()
            self.board.draw()
            print("ROUND #%s: " % str(roundNumber).zfill(roundNumberZeroFill), end="")
            playerCount = len(self.players)
            # Share fences between players
            playerFenceCount = int(self.totalFenceCount/playerCount)
            self.board.fences, self.board.pawns = [], []
            # For each player
            for i in range(playerCount):
                player = self.players[i]
                # Place player pawn at start position and add fences to player stock
                player.pawn.place(player.startPosition)
                for j in range(playerFenceCount):
                    player.fences.append(Fence(self.board, player))
            # Define randomly first player (coin toss)
            currentPlayerIndex = random.randrange(playerCount)
            finished = False
            while not finished:
                player = self.players[currentPlayerIndex]
                # The player chooses its action (manually for human players or automatically for bots)
                action = player.play(self.board)
                if isinstance(action, PawnMove):
                    player.movePawn(action.toCoord)
                    # Check if the pawn has reach one of the player targets
                    if player.hasWon():
                        finished = True
                        print("Player %s won" % player.name)
                        player.score += 1
                elif isinstance(action, FencePlacing):
                    player.placeFence(action.coord, action.direction)
                elif isinstance(action, Quit):
                    finished = True
                    print("Player %s quitted" % player.name)
                currentPlayerIndex = (currentPlayerIndex + 1) % playerCount
                if INTERFACE:
                    time.sleep(TEMPO_SEC)
        print()
        #self.board.drawOnConsole()
        # Display final scores
        print("FINAL SCORES: ")
        bestPlayer = self.players[0]
        for player in self.players:
            print("- %s: %d" % (str(player), player.score))
            if player.score > bestPlayer.score:
            	bestPlayer = player
        print("Player %s won with %d victories!" % (bestPlayer.name, bestPlayer.score))

    def placePawnSim(self, coord, board, pawn, ind=-1):
        fromCoord, toCoord = None if coord is None else coord.clone(), coord
        pawn.coord = coord
        board.pawns.append(pawn)
        if(ind==0):
            board.pawn_0 = pawn
        if(ind==1):
            board.pawn_1 = pawn
        board.updateStoredValidActionsAfterPawnMove(fromCoord, toCoord)

    def placeFenceSim(self, coord, direction, board):
        fence = Fence(board, self)
        fence.coord = coord
        fence.direction = direction
        board.fences.append(fence)
        board.updateStoredValidActionsAfterFencePlacing(coord, direction)
    def end(self):
        """
        Called at the end in order to close the window.
        """
        if INTERFACE:
            self.board.window.close()

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        #mcts_bot = MCTS_Bot.MCTS_Bot()
        board = self.board
        board.pawn_0 = Pawn(board, MCTS_Bot)
        board.pawn_1 = Pawn(board, MCTS_Bot)
        print("111: " + str(board.cols))
        board.initStoredValidActions()
        self.placePawnSim(board.startPosition(0), board, board.pawn_0, ind=0)
        self.placePawnSim(board.startPosition(1), board, board.pawn_1, ind=1)
        #self.placePawnSim(self.players[0].startPosition(0), board, self.players[0].pawn, ind=0)
        #self.placePawnSim(self.players[1].startPosition(1), board, self.players[1].pawn, ind=1)
        
       # playerCount = 2
        #print("LEN: " + str(len(self.players)))
       # playerFenceCount = int(self.totalFenceCount/playerCount)
        #for i in range(playerCount):
           # player = board.players[i]
            # Place player pawn at start position and add fences to player stock
           # for j in range(playerFenceCount):
                #player.fences.append(Fence(self.board, player))
       # playerCount = 2
       # playerFenceCount = int(self.totalFenceCount/playerCount)
        #for i in range(2):
          #  if(i==0):
                #player = self.players[0]
            #    player = board.pawn_0.player
          #  else:
               # player = self.players[1]
              #  player = board.pawn_1.player
         #   for j in range(playerFenceCount):
             #   player.fences.append(Fence(board, player ))
        #playerCount = len(self.players)
        
        return board
       
    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.board.cols, self.board.rows)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 4+(2*(self.board.cols-1)**2)

    def getActionSpace(self, board):
        action_space = []
        size = 4+(2*(self.cols-1)**2)
        actions = board.storedValidPawnMoves + board.storedValidFencePlacings
        if len(actions) == size:
            for i in range(size):
                action_space.append((i, actions[i]))
        else:
            print("SIZE OF ACTION SPACE: " + str(len(actions)))
            print("ERROR")
        return action_space
        
        

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player
        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        if(player==0):
            if isinstance(action, PawnMove):
                self.placePawnSim(action.toCoord, board, board.pawn_0)
            else:
                self.placeFenceSim(action.coord, action.direction, board)
            return board, 1
        else:
            if isinstance(action, PawnMove):
                self.placePawnSim(action.toCoord, board, board.pawn_1)
            else:
                self.placeFenceSim(action.coord, action.direction, board)
            return board, 0

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player
        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        if(player==0):
            coord = board.pawn_0.coord
            return board.storedValidFencePlacings + board.storedValidPawnMoves[coord]
        else:
            coord = board.pawn_1.coord
            return board.storedValidFencePlacings + board.storedValidPawnMoves[coord]


    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        print("CHRIOS: " + str(board.pawn_0.player))
        if(player==0):
            if(board.pawn_0.player().hasWon()):
                return 1
            elif(board.pawn_1.player().hasWon()):
                return -1
            else:
                return 0
        if(player==1):
            print("JUST: " + str(board.pawn_1.player))
            if(board.pawn_1.player().hasWon()):
                return 1
            elif(board.pawn_0.player().hasWon()):
                return -1
            else:
                return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        print("1: " + str(board.pawn_0.coord) )
        return board
        return [[board.pawn_0.coord.col, board.pawn_0.coord.row],
                            [board.pawn_1.coord.col, board.pawn_1.coord.row],
                            board.fences]
       # torch.tensor([[board.pawn_0.coord.col, board.pawn_0.coord.row],
                           # [board.pawn_1.coord.col, board.pawn_1.coord.row],
                           # board.fences])
        #pass
        
       # if(player==-1):
       #     return board
       # if(player==1):
        #    return board

    def getTensorForm(self, board):
        def pad_to_dense(M):
            """Appends the minimal required amount of zeroes at the end of each 
        array in the jagged array `M`, such that `M` looses its jagedness."""

            maxlen = max(len(r) for r in M)
            Z = np.zeros((len(M), maxlen))
            for enu, row in enumerate(M):
                Z[enu, :len(row)] += row 
            return Z
        
        return pad_to_dense([[board.pawn_0.coord.col, board.pawn_0.coord.row],
                            [board.pawn_1.coord.col, board.pawn_1.coord.row],
                            board.fences])

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()
        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board
        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        #pass
        return str(board.pawn_0.coord) + ';'+ str(board.pawn_1.coord) + ';' + ''.join(board.fences)
       # return ''.join([[board.pawn_0.coord],
               #[board.pawn_1.coord],
               # board.fences])
