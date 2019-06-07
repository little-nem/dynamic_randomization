import numpy as np
import random
import gym

from environment import RandomizedEnvironment
from agent import Agent
from replay_buffer import Episode

import fetch_slide_2

experiment="FetchSlide2-v1"

MODEL_NAME = "checkpoints/ckpt_episode_0"
ROLLOUT_NUMBER = 100
BATCH_SIZE = 32
MAX_STEPS = 50


RENDER = True

# initialize the agent, both the actor/critic (and target counterparts) networks
agent = Agent(experiment, BATCH_SIZE*MAX_STEPS)

# initialize the environment sampler
randomized_environment = RandomizedEnvironment(experiment, [0.0, 1.0], [])

# load the wanted model
agent.load_model(MODEL_NAME)

success_number = 0

randomized_environment.sample_env()
env, env_params = randomized_environment.get_env()

for test_ep in range(ROLLOUT_NUMBER):
    print("Episode {}".format(test_ep))


    current_obs_dict = env.reset()

    # read the current goal, and initialize the episode
    goal = current_obs_dict['desired_goal']
    episode = Episode(goal, env_params, MAX_STEPS)

    # get the first observation and first fake "old-action"
    # TODO: decide if this fake action should be zero or random
    obs = current_obs_dict['observation']
    achieved = current_obs_dict['achieved_goal']
    last_action = env.action_space.sample()

    episode.add_step(last_action, obs, 0, achieved)

    done = False

    # rollout the whole episode
    while not done:
        obs = current_obs_dict['observation']
        history = episode.get_history()

        if RENDER: env.render()
        action = agent.evaluate_actor(agent._actor.predict_target, obs, goal, history)

        new_obs_dict, step_reward, done, info = env.step(action[0])

        new_obs = new_obs_dict['observation']
        achieved = new_obs_dict['achieved_goal']

        episode.add_step(action[0], new_obs, step_reward, achieved, terminal=done)

        current_obs_dict = new_obs_dict

    if info['is_success'] > 0.0:
        success_number += 1

    #randomized_environment.close_env()

print("Success rate : {}".format(success_number/ROLLOUT_NUMBER))
