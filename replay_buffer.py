from collections import deque
import random
import numpy as np

class SamplingSizeError(Exception):
    pass

class Episode:
    def __init__(self, goal, env, max_history_timesteps):
        self._states = []
        self._actions = []
        self._rewards = []
        self._terminal = []

        # numpy array that can be used to directly feed the network
        self._history = []
        self._dim_history_atom = 0

        self._max_history_timesteps = max_history_timesteps

        self._goal = goal
        self._env = env


    def add_step(self, action, obs, reward, terminal = False):
        self._actions.append(action)
        self._states.append(obs)
        self._rewards.append(reward)

        # if the history is empty, initialize it using the dims of the action
        # and state provided as arguments
        if self._history == []:
            self._dim_history_atom = action.shape[0] + obs.shape[0]
            self._history = np.array(self._max_history_timesteps*[np.zeros(self._dim_history_atom)])

        self._history = np.append(self._history, [np.concatenate((action, obs))], axis = 0)[1:]
        self._terminal.append(terminal)

    def get_history(self, t = -1):
        # returns the history for calling the actor at step t (if t == -1, return current history)
        # (ie. history = [a_(t - max_history_timesteps), o_(t - max_history_timesteps), ...,
        # a_(t-1), o_(t-1)]
        # zero-padded to use with the lstm
        if t == -1:
            return self._history
        else:
            history = np.array(self._max_history_timesteps*[np.zeros(self._dim_history_atom)])

            for step in range(max(t - self._max_history_timesteps, 0), t):
                action = self._actions[step]
                obs = self._states[step]

                # potential speedup only rewriting the good line instead of creating a new array
                history = np.append(history, [np.concatenate((action, obs))], axis = 0)[1:]

            return history

    def get_goal(self):
        return self._goal

    def get_terminal(self):
        return self._terminal

    def get_actions(self):
        return self._actions

    def get_states(self):
        return self._states

    def get_rewards(self):
        return self._rewards

    def get_env(self):
        return self._env

class ReplayBuffer:
    def __init__(self, buffer_size, random_seed = 0):
        self._buffer_size = buffer_size
        self._buffer = deque()
        self._current_count = 0
        random.seed(random_seed)

    def size(self):
        return self._current_count

    def add(self, episode):
        if self._current_count >= self._buffer_size:
            self._buffer.popleft()
            self._current_count -= 1

        self._buffer.append(episode)
        self._current_count += 1

    def sample_batch(self, batch_size):
        if batch_size > self._current_count:
            raise SamplingSizeError

        return random.sample(self._buffer, batch_size)
     
