from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque

class DQN:

    def __init__(self, action_space, state_space, lr=.001, gamma=.95, batch_size=64, epsilon=1, epsilon_min=.01, 
                epsilon_decay=.999, memory=100000, fc1_dims=128, fc2_dims=64):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        self.memory = deque(maxlen=memory)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model = self.build_model(fc1_dims, fc2_dims)


    def build_model(self, fc1_dims, fc2_dims):

        model = Sequential()
        model.add(Dense(fc1_dims, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(fc2_dims, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
