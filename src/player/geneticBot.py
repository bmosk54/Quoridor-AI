from src.player.BuilderBot import *
from src.player.RunnerBot  import *
from src.interface.Board import *
from src.interface.Fence import *
from src.Path import *
import numpy

class geneticBot(BuilderBot, RunnerBot):

    chromosome = numpy.zeros(4)

    def setChromosome(self, chromosome):
        self.chromosome = chromosome

    def eval(self, board, ind, pawn1, pawn2):
        p1 = ind
        p2 = 1 - ind

        mht1 = Path.ManhattanDistance(pawn1.coord, board.endPositions(p1))
        mht2 = Path.ManhattanDistance(pawn2.coord, board.endPositions(p2))
        bfs1 = len(Path.BreadthFirstSearch(board, pawn1.coord, board.endPositions(p1), False).moves)
        bfs2 = len(Path.BreadthFirstSearch(board, pawn2.coord, board.endPositions(p2), False).moves)

        return (self.chromosome[0]*mht1+self.chromosome[1]*mht2\
               +self.chromosome[2]*bfs1+self.chromosome[3]*bfs2)

    def placePawnSim(self, coord, board, pawn):
        fromCoord, toCoord = None if coord is None else coord.clone(), coord
        pawn.coord = coord
        board.pawns.append(pawn)
        board.updateStoredValidActionsAfterPawnMove(fromCoord, toCoord)

    def placeFenceSim(self, coord, direction, board):
        fence = Fence(board, self)
        fence.coord = coord
        fence.direction = direction
        board.fences.append(fence)
        board.updateStoredValidActionsAfterFencePlacing(coord, direction)

    def moveAlongTheShortestPathPawn(self, board, pawn, ind_of_pawn) -> IAction:
        path = Path.BreadthFirstSearch(board, pawn.coord, board.endPositions(ind_of_pawn), ignorePawns=False)
        if path is None:
            path = Path.BreadthFirstSearch(board, pawn.coord, board.endPositions(ind_of_pawn), ignorePawns=True)
            firstMove = path.firstMove()
            if not board.isValidPawnMove(firstMove.fromCoord, firstMove.toCoord, ignorePawns=False):
                # board.drawOnConsole()
                return None
        return path.firstMove()

    def computeFencePlacingImpactsFence(self, board, pawns, inds, curr_pawn):
        fencePlacingImpacts = {}
        # Compute impact of every valid fence placing
        for fencePlacing in board.storedValidFencePlacings:
            try:
                impact = board.getFencePlacingImpactOnPathsPawn(fencePlacing, pawns, inds)
            # Ignore path if it is blocking a player
            except PlayerPathObstructedException as e:
                continue
            globalImpact = 0
            for pawn in impact:
                globalImpact += (-1 if pawn == curr_pawn else 1) * impact[pawn]
            fencePlacingImpacts[fencePlacing] = globalImpact
        return fencePlacingImpacts

    def minimaxValue(self, board, isMax, depth, ind, pawn1, pawn2):
        if depth == 0:
            return self.eval(board, ind, pawn1, pawn2)
        if isMax:
            curr_pawn = pawn1
            pawn_moves = [self.moveAlongTheShortestPathPawn(board, curr_pawn, ind)]
            fencePlacingImpacts = self.computeFencePlacingImpactsFence(board, [pawn1, pawn2], [ind, 1 - ind], curr_pawn)
            if len(fencePlacingImpacts) < 1:
                fence_moves = []
            else:
                fence_moves = [self.getFencePlacingWithTheHighestImpact(fencePlacingImpacts)]
            moves = pawn_moves + fence_moves
            rets = []
            for move in moves:
                curr_board = Board(board=board)
                if isinstance(move, PawnMove):
                    self.placePawnSim(move.toCoord, curr_board, curr_pawn)
                    rets.append(self.minimaxValue(curr_board, not isMax, depth - 1, ind, pawn1, pawn2))
                else:
                    self.placeFenceSim(move.coord, move.direction, curr_board)
                    rets.append(self.minimaxValue(curr_board, not isMax, depth - 1, ind, pawn1, pawn2))

            return max(rets)

        else:
            curr_pawn = pawn2
            pawn_moves = [self.moveAlongTheShortestPathPawn(board, curr_pawn, ind)]
            fencePlacingImpacts = self.computeFencePlacingImpactsFence(board, [pawn1, pawn2], [ind, 1 - ind], curr_pawn)
            if len(fencePlacingImpacts) < 1:
                fence_moves = []
            else:
                fence_moves = [self.getFencePlacingWithTheHighestImpact(fencePlacingImpacts)]
            moves = pawn_moves + fence_moves
            rets = []
            for move in moves:
                curr_board = Board(board=board)
                if isinstance(move, PawnMove):
                    self.placePawnSim(move.toCoord, curr_board, curr_pawn)
                    rets.append(self.minimaxValue(curr_board, not isMax, depth - 1, ind, pawn1, pawn2))
                else:
                    self.placeFenceSim(move.coord, move.direction, curr_board)
                    rets.append(self.minimaxValue(curr_board, not isMax, depth - 1, ind, pawn1, pawn2))

            return min(rets)

    def play(self, board) -> IAction:
        ind = board.pawns.index(self.pawn)
        pawn1 = board.pawns[ind]
        pawn2 = board.pawns[1 - ind]

        if self.remainingFences() < 1 or len(board.storedValidFencePlacings) < 1:
            return self.moveAlongTheShortestPath(board)

        fencePlacingImpacts = self.computeFencePlacingImpacts(board)
        # If no valid fence placing, move pawn
        if len(fencePlacingImpacts) < 1:
            return self.moveAlongTheShortestPath(board)

        pawn_moves = [self.moveAlongTheShortestPath(board)]
        fencePlacingImpacts = self.computeFencePlacingImpacts(board)
        fence_moves = [self.getFencePlacingWithTheHighestImpact(fencePlacingImpacts)]
        moves = pawn_moves + fence_moves
        action = None
        hi = float('-inf')
        for move in moves:
            temp = self.minimaxValue(board, True, 1, ind, pawn1, pawn2)
            if temp > hi:
                hi = temp
                action = move

        return move