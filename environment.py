import gym
import numpy as np
import random

import fetch_slide_2

# hardcoded for fetch_slide_2, with only friction
class RandomizedEnvironment:
    """ Randomized environment class """
    def __init__(self, experiment, parameter_ranges, goal_range):
        self._experiment = experiment
        self._parameter_ranges = parameter_ranges
        self._goal_range = goal_range
        self._params = [0]

    def sample_env(self):
        mini = self._parameter_ranges[0]
        maxi = self._parameter_ranges[1]
        pick = mini + (maxi - mini)*random.random()

        self._params = np.array([pick])
        self._env = gym.make(self._experiment)
        self._env.env.reward_type="dense"
        self._env.set_property('object0', 'geom_friction', [pick])
    def get_env(self):
        """
            Returns a randomized environment and the vector of the parameter
            space that corresponds to this very instance
        """
        return self._env, self._params

    def close_env(self):
        self._env.close()

    def get_goal(self):
        return

