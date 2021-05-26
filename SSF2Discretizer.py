import gym
import numpy as np
import retro


"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos

Most of this is taken from Retro documentation, and adopted to our own button needs
"""

import gym
import numpy as np
import retro

combo_list = [['SELECT'], ['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], \
         ['UP'], ['RIGHT', 'UP'], ['LEFT', 'UP'], ['Y'], ['RIGHT', 'Y'], ['LEFT', 'Y'], ['DOWN', 'Y'], \
              ['X'], ['RIGHT', 'X'], ['LEFT', 'X'], ['DOWN', 'X'], \
                ['L'], ['RIGHT', 'L'], ['LEFT', 'L'], ['DOWN', 'L'],
                ['B'], ['RIGHT', 'B'], ['LEFT', 'B'], ['DOWN', 'B'],
                ['A'], ['RIGHT', 'A'], ['LEFT', 'A'], ['DOWN', 'A'],
                ['R'], ['RIGHT', 'R'], ['LEFT', 'R'], ['DOWN', 'R'], ['Y', 'B']]

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class SSF2Discretizer(Discretizer):
    """
    Based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=combo_list)
        

    def getAction(action_ls):
        for i in range(len(combo_list)):
            flag = True
            if len(action_ls) == len(combo_list[i]):
                for j in range(len(combo_list[i])):
                    if action_ls[j] != combo_list[i][j]:
                        flag = False
                if flag:
                    return i
        return -1

    def getActionFromIndex(i):
        return combo_list[i]