import numpy as np

class QLearning:
    def __init__(self, n_actions, n_states, discount=0.9, alpha = 0.01, epsilon=0.1, decay = 0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.Q = np.zeros([n_states, n_actions])
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.count = 0
        self.decay = decay
        self.prev_state = -1
        self.prev_action = -1
    def act(self):
        ## by default, act greedily
        if (np.random.uniform() < self.epsilon):
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[self.state, :])
    
    def update(self, action, reward, state):
        self.prev_action = action
        if (self.prev_state >=0):
            self.Q[self.prev_state, self.prev_action] += self.alpha * (reward + self.discount * max(self.Q[state, :]) - self.Q[self.prev_state, self.prev_action])
        self.prev_state = state

    def reset(self, state):
        self.state = state
        self.prev_state = -1
        
