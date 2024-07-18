## Import the GYM API
import gym
import gym_bandits
from gym import spaces
from gym.utils import seeding

import numpy as np

## This defines the Chain environment
class Chain(gym.Env):
  def __init__(self, states=5, delta=0.4, epsilon=0.1):
    self.n_states = states
    self.n_actions = 2
    self.r_dist = np.zeros(states)
    self.r_dist[0] = epsilon
    self.r_dist[self.n_states - 1] = 1
    self.action_space = spaces.Discrete(self.n_actions)
    self.observation_space = spaces.Discrete(self.n_states)
    self.delta = delta
    self.epsilon = epsilon
    self._seed()
    self.reset()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    assert self.action_space.contains(action)
    done = False
    reward = np.random.binomial(1, self.r_dist[self.state])

    move = action
    ## swap the move with a delta probability
    if (np.random.uniform()<self.delta):
        move = 1 - action
    if (move==0):
      self.state = 0
    else:
      self.state += 1
      if (self.state > self.n_states  - 1):
        self.state = self.n_states - 1
    return self.state, reward, done, {}

  def reset(self):
    self.state = 0
    return 0

  def render(self, mode='human', close=False):
    pass



 


