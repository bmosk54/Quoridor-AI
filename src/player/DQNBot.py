from src.player.IBot     import *
from src.action.IAction  import *
from src.Path            import *
from src.player.DQN      import *
from src.interface.Board import *


class DQNBot(IBot):
    def train_dqn(episode):
        moves = self.validPawnMoves(self.pawn.coord)
        moves += self.validFencePlacings()
        loss = []

        action_space = len(moves)
        state_space = 5
        max_steps = 10000

        agent = DQN(action_space, state_space)
        for e in range(episode):
            state = env.reset()
            state = np.reshape(state, (1, state_space))
            score = 0
            for i in range(max_steps):
                action = agent.act(state)
                reward, next_state, done = env.step(action)
                score += reward
                next_state = np.reshape(next_state, (1, state_space))
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                agent.replay()
                if done:
                    print("episode: {}/{}, score: {}".format(e, episode, score))
                    break
            loss.append(score)
        return loss

    def play(self, board):
        