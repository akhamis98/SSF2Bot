import numpy as np

class RandomAgent():
    
    def __init__(self, observation_space, action_space):
        self.actions = action_space

    def __call__(self, obs):
        return np.random.randint(self.actions)


class Agent():

    def __init__(observation_space, action_space):
        self.actions = action_space
        self.states = observation_space

        self.policy = np.zeros((self.states,), dtype=np.int)
        self.value  = np.zeros((self.states,), dtype=np.int)

    def policy_evaluation(self, transitions, rewards):
        pass

    def policy_improvement(self):
        pass

    def policy_iteration(self):
        pass

    def value_iteration(self):
        pass

    def __call__(self, obs):
        return self.policy[obs]

