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


class MCTS_Bot(IBot):
    def play(self, board) -> IAction:
        pass