# This is a LearnerBot working as an agent that interacts and learns for the deep Q-network.
#
#
# Author: Xiaoyu Liu
#
#
# Define some basic components in this task:
#  1. state: a n*n*2+2 vector where n is the length of the board
#     for a 5 x 5 board, state vector is of dimension 52 ,
#     we store the grid for PAWNs and the grid for FENCES in different coordinate systems
#
#     5 possible values:
#
#     0-nothing
#     1-player1
#     2-player2
#     3-fence from player 1
#     4-fence from player 2
#
#     two remaining vector records fence left for player 1 and 2
#
#
#
#
#  2. action 4-dimensional vector
#
#     [1] move or not  0 - not move 1-4 direction
#
#     [2] fence coordinates [x,y] and direction [Horizontal or Vertical]
#
#     if we choose not to place fence, we fill the vector with -1s.
#
#
# 3. reward
#
#    [1] If the player gets closer (by distance) to the destination, we have a +1 reward
#    I choose a very direct reward measurement as just the euclidean distance between the target
#    the baseline.
#

import numpy as np
import random
from collections import namedtuple, deque



import torch
import torch.optim as optim

from src.DQN.QNetwork import *
from src.DQN.utils import *

from src.player.IBot    import *
from src.action.IAction import *

BUFFER_SIZE = int(1e5)  #replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LearnerBot(IBot):
    """
    LearnerBot talks to the environment, obtaining current state and pass future actions,

    Meanwhile it helps learns the deep Q-Network.

    """

    def __init__(self, grid_size, board, action_dim=4):

        """Initialize an Agent object.

        Params
        =======
            1. grid_size (int): length of the board, we use grid_size to calculate the state_dimension

               state_dim = grid_size * grid_size + 2 (fences left in both players)

            2. action_dim (int): dimension of each action

        """

        self.state_size = grid_size * grid_size + 2
        self.action_dim = action_dim
        self.board = board

        # Initializing two Q-Networks
        self.qnetwork_local = QNetwork(state_dim, action_dim).to(device)
        self.qnetwork_target = QNetwork(state_dim, action_dim).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_step, done):

        """


        """

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def play(self, state, eps=0)-> IAction:

        """Returns action for given state as per current policy

        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon -greedy action selction
        if random.random() > eps:
            x,y = recover_coords(5, state, "player1").row, recover_coords(5, state, "player1").col
            next_action =  action_to_move(action_values,x,y)
            if legal(next_action):
                return next_action

        if random.randint(0, 2) == 0 and self.remainingFences() > 0 and len(board.storedValidFencePlacings) > 0:
            return self.placeFenceRandomly(self.board)
        else:
            return self.moveRandomly(self.board)

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones = experiences

        # compute and minimize the loss
        criterion = torch.nn.MSELoss()

        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()

        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()

        # shape of output from the model (batch_size, action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    # copy from randombot
    def moveRandomly(self, board) -> IAction:
        validPawnMoves = board.storedValidPawnMoves[self.pawn.coord] #board.validPawnMoves(self.pawn.coord)
        return random.choice(validPawnMoves)

    def placeFenceRandomly(self, board) -> IAction:
        randomFencePlacing = random.choice(board.storedValidFencePlacings)
        attempts = 5
        while board.isFencePlacingBlocking(randomFencePlacing) and attempts > 0:
            #print("Cannot place blocking %s" % randomFencePlacing)
            randomFencePlacing = random.choice(board.storedValidFencePlacings)
            attempts -= 1
        if (attempts == 0):
            return self.moveRandomly()
        return randomFencePlacing


class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""

    def __init__(self, action_dim, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_dim (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.action_dim = action_dim
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state",
                                                                 "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)