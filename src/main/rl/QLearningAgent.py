from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
from collections import deque
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, epsilon = 1.0, epsilon_decay = 0.995, epsilon_min = 0.1 ):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_nn(self.state_size, self.action_size) # represents the value function
        self.epsilon = epsilon # Agent explores with probability epsilon and takes best action with probability 1 - epsilon -> epsilon greedy
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_rate = 0.99 # discount rate of future rewards
        self.memory = deque(maxlen=3000) # [(state, action, reward, next_state, done)...]


    def _build_nn(self, state_size, action_size) -> Sequential:

        model = Sequential()
        learning_rate = 0.01

        # Input layer
        model.add(Dense(20, input_dim=state_size, activation='relu'))
        # hidden layer
        model.add(Dense(20, activation='relu'))
        # Output layer with # of actions
        model.add(Dense(action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate))

        print(model.summary())

        return model

    def choose_action(self, state) -> int:
        if self._want_to_explore():
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        best_action_value = np.argmax(action_values[0])
        return best_action_value

    def _want_to_explore(self) -> bool:
        threshold = max(self.epsilon, self.epsilon_min)
        return np.random.rand() >= threshold

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn_from_past(self, batch_size):

        batch = self._build_batch(batch_size)

        print("Training on batch: " +str(sorted([round(ep[2], 4) for ep in batch[:20]])[::-1]))

        for state, action, reward, next_state, finished in batch:
            if not finished:
                value = self.predict_value(reward, state)
            else:
                value = reward

            target_f = self.model.predict(state)
            target_f[0][action] = value
            self.model.fit(state, target_f, epochs=1, verbose=0)

        self.epsilon *= self.epsilon_decay

    def predict_value(self, reward, state) -> float:
        value = (reward + self.discount_rate * np.amax(self.model.predict(state)[0]))
        return value

    def load_model_weights(self, name):
        self.model.load_weights(name)

    def save_model_weights(self, name):
        self.model.save_weights(name)

    def _build_batch(self, batch_size) -> list:
        minibatch =list(self.memory)[: -batch_size]

        return minibatch
