from src.player.MCTS_Bot import *
from src.player.MCTS_Bot import MCTS_Bot as Mcts_bot
#from src.QuoridorGame import QuoridorGame as Game
from src.Game import *
from src.player.mcts.NNet import NNetWrapper as nn
from utils import *
import getopt
PARAMETERS_ERROR_RETURN_CODE = 1

from src.Settings              import *
from src.player.Human          import *
from src.player.RandomBot      import *
from src.player.RunnerBot      import *
from src.player.BuilderBot     import *
from src.player.BuildAndRunBot import *

args = dotdict({
    'numIters': 10, #1000
    'numEps': 50, #100
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','5x5best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


def printUsage():
    print("Usage: python quoridor.py [{-h|--help}] {-p|--players=}<PlayerName:PlayerType,...> [{-r|--rounds=}<roundCount>] [{-x|--cols=}<ColCount>] [{-y|--rows=}<RowCount>] [{-f|--fences=}<TotalFenceCount>] [{s|--square_size=}<SquareSizeInPixels>]")
    print("Example: python quoridor.py --players=Alain:Human,Benoit:BuilderBot,Caroline:RandomBot,Daniel:RunnerBot --square-size=32")

def readArguments():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:r:w:x:y:s:h", ["players=", "rounds=", "cols=", "rows=", "fences=", "square_size=", "help"])
    except getopt.GetoptError as err:
        print(err)
        printUsage()
        sys.exit(PARAMETERS_ERROR_RETURN_CODE)
    players = []
    rounds = 1
    cols = 9
    rows = 9
    totalFenceCount = 20
    squareSize = 32
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            printUsage()
            sys.exit(0)
        elif opt in ("-p", "--players"):
            for playerData in arg.split(","):
                playerName, playerType = playerData.split(":")
                if playerType not in globals():
                    print("Unknown player type %s . Abort." % (playerType))
                    sys.exit(PARAMETERS_ERROR_RETURN_CODE)
                players.append(globals()[playerType](playerName))
            if len(players) not in (2, 4):
                print("Expect 2 or 4 players. Abort.")
                sys.exit(PARAMETERS_ERROR_RETURN_CODE)
        elif opt in ("-r", "--rounds"):
            rounds = int(arg)
        elif opt in ("-x", "--cols"):
            cols = int(arg)
        elif opt in ("-y", "--rows"):
            rows = int(arg)
        elif opt in ("-f", "--fences"):
            totalFenceCount = int(arg)
        elif opt in ("-s", "--square_size"):
            squareSize = int(arg)
        else:
            print("Unhandeld option. Abort.")
            sys.exit(PARAMETERS_ERROR_RETURN_CODE)
    return players, rounds, cols, rows, totalFenceCount, squareSize

if __name__=="__main__":
    #g = Game(5)
    players, rounds, cols, rows, totalFenceCount, squareSize = readArguments()
    #players = [MCTS_Bot, MCTS_Bot]
    game = Game(players, cols, rows, totalFenceCount, squareSize)
    
    nnet = nn(game)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Mcts_bot(game, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.train_mcts()