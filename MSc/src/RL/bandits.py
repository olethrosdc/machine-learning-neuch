from abc import ABC

# pip install gymnasium==0.27.1
import gymnasium as gym

from gymnasium import spaces
from gymnasium.utils import seeding

import matplotlib.pyplot as plt
import numpy as np


# Smooth down the values, to better visualize improvement
def moving_average(x, K):
    T = x.shape[0]
    n = x.shape[1]
    m = int(np.ceil(T / K))
    y = np.zeros([m, n])
    for alg in range(n):
        for t in range(m):
            y[t, alg] = np.mean(x[t * K:(t + 1) * K, alg])
    return y


## Here bandit problems are sampled from a Beta distribution
class BetaBandits(gym.Env):
    def __init__(self, bandits=10, alpha=1, beta=1):
        self.r_dist = np.zeros(bandits)
        for i in range(bandits):
            self.r_dist[i] = np.random.beta(alpha, beta)
        self.n_bandits = bandits
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)

        # Actual best arm (unknown to the algorithm)
        self.best_arm_param = np.max(self.r_dist)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = True
        reward = np.random.binomial(1, self.r_dist[action])
        return 0, reward, done, {}

    def reset(self):
        return 0


class BanditAlgorithm(ABC):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def act(self) -> int:
        """
        Choose an arm/action
        """
        pass

    def update(self, action, reward):
        """
        Update our strategy
        """
        pass


class EpsilonGreedy(BanditAlgorithm):
    def __init__(self, n_actions):
        super().__init__(n_actions)

        ...

    def act(self):
        pass

    def update(self, action, reward):
        pass


class ThompsonSampling(BanditAlgorithm):
    def __init__(self, n_actions):
        super().__init__(n_actions)
        
    def act(self):
        pass

    def update(self, action, reward):
        pass



# (At home)
class UCB(BanditAlgorithm):
    def __init__(self, n_actions):
        super().__init__(n_actions)

    def act(self):
        pass

    def update(self, action, reward):
        pass

# We can play with this
n_actions = 2

n_experiments = 30
T = 10000
environments = []

# Instantiate some bandit problems
for experiment in range(n_experiments):
    environments.append(BetaBandits(n_actions, 1, 1))

# The algorithms we want to benchmark
algs = [EpsilonGreedy]

n_algs = len(algs)
reward_t = np.zeros((T, n_algs))
regret_t = np.zeros((T, n_algs))

total_reward = np.zeros(n_algs)
for experiment in range(n_experiments):
    env = environments[experiment]
    for alg_index, Alg in enumerate(algs):
        alg = Alg(n_actions)
        run_reward = 0
        for i_episode in range(T):

            # An episode lasts one step
            env.reset()

            # we choose an arm (action)
            action = alg.act()

            # Observe the reward for choosing the action
            reward = env.step(action)  # play the action in the environment

            # learn
            alg.update(action, reward)

            run_reward += reward
            reward_t[i_episode, alg_index] += reward
            regret_t[i_episode, alg_index] += env.best_arm_param - reward

        total_reward[alg_index] += run_reward
        env.close()

total_reward /= n_experiments
reward_t /= n_experiments
regret_t /= n_experiments

cummulative_regret = np.cumsum(regret_t, axis=0)

plt.plot(moving_average(reward_t, 10))
plt.legend([c.__name__ for c in algs])
plt.ylabel("Average reward")
plt.savefig("benchmark_reward.pdf")
plt.clf()

plt.plot(moving_average(cummulative_regret, 10))
plt.legend([c.__name__ for c in algs])
plt.ylabel("Cumulative regret")
plt.title("Total algorithm regret")
plt.savefig("benchmark_regret.pdf")
