from src.player.IBot    import IBot
from src.action.IAction import * 
#from src.player.mcts.MCTS    import MCTS
from src.interface.Board import *
from src.player.tree import MCTS
import numpy as np
from src.player.tree.pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from src.player.tree.NNet import NNetWrapper as nn
from utils import *
from collections import deque
#from src.Game import Game

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
    'load_model': True,
    'load_folder_file': ('./temp','5x5best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})
 

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    #def __init__(self, game, nnet, args):
    def __init__(self, game, nnet, args=args):
        IPlayer.__init__(self)
        #self.fences = []
        self.game = game
        #cols = 9, rows = 9, totalFenceCount = 20, squareSize = 32, innerSize = None
        #print("CALL1")
        board = Board(self, cols=9, rows=9, squareSize=32, innerSize=3)
       
       # self.nnet = nn(board)
        self.nnet = nnet
        
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        #self.pnet = self.nnet.__class__(board)
        self.args = args
        self.mcts = MCTS.MCTS(self.game, self.nnet, self.args)
       # self.mcts = MCTS.MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()
    def train_mcts(self):
        self.learn();
    def play1(self, board): #-> IAction:
        # If no fence left, move pawn
        if self.remainingFences() < 1 or len(board.storedValidFencePlacings) < 1:
            return self.moveRandomly(board)
        fencePlacingImpacts = self.computeFencePlacingImpacts(board)
        # If no valid fence placing, move pawn
        if len(fencePlacingImpacts) < 1:
            return self.moveRandomly(board)
        # Choose fence placing with the greatest impact
        bestFencePlacing = self.getFencePlacingWithTheHighestImpact(fencePlacingImpacts)
        # If impact is not positive, move pawn
        if fencePlacingImpacts[bestFencePlacing] < 1:
            return self.moveRandomly(board)
        return bestFencePlacing
    #def play(self, board):
     #   self.train_mcts()


    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        
       

        #board = Game()
        self.curPlayer = 1
       # self.curPlayer = board.pawns.index()
        episodeStep = 0
        while True and episodeStep<200:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            print("TYPE: " + str(type(canonicalBoard)))
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)

            if np.sum(pi) == 0: break

            #sym = self.game.getSymmetries(canonicalBoard, pi)
            #for b,p in sym:
            #    trainExamples.append([b, self.curPlayer, p, None])
            #self.game.print_board(canonicalBoard)

            action = np.random.choice(len(pi), p=pi)
            trainExamples.append([canonicalBoard, self.curPlayer, pi, None])
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*x[1]) for x in trainExamples]
        #return [(x[0],x[2],0) for x in trainExamples]
        return []

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    self.mcts = MCTS.MCTS(self.game, self.nnet, self.args)   # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)
                trainStats = [0,0,0]
                for _,_,res in iterationTrainExamples:
                    trainStats[res] += 1
                print(trainStats)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i-1)

            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
