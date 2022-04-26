import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Define a Q-Network

    """

    def __init__(self, state_dim, action_dim, fc1_dim=64, fc2_dim=64):
        """
        state_dim: dimension of the state space

        action_dim: dimension of the action space

        fc1_dim: dimension of the first fully connected layer

        fc2_dim: dimension of the second fully connected layer

        """

        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, action_dim)

    def forward(self, x):
        """
        input x: current state

        output: action to take

        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)