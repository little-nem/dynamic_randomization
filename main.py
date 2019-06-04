import numpy as np
import random
import gym

from environment import RandomizedEnvironment
from agent import Agent
from replay_buffer import Episode, ReplayBuffer

EPISODES = 1000

experiment = "FetchReach-v1"
env = gym.make(experiment)

# Program hyperparameters
TESTING_INTERVAL = 50 # number of updates between two evaluation of the policy
TESTING_ROLLOUTS = 100 # number of rollouts performed to evaluate the current policy

# Algorithm hyperparameters
BATCH_SIZE = 32
BUFFER_SIZE = 100000
MAX_STEPS = 50 # WARNING: defined in multiple files...
GAMMA = 0.99
K = 0.8 # probability of replay with H.E.R.

# Initialize the agent, both the actor/critic (and target counterparts) networks
agent = Agent(experiment, BATCH_SIZE*MAX_STEPS)

# Initialize the environment sampler
randomized_environment = RandomizedEnvironment(experiment, [], [])

# Initialize the replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE)

# should be done per episode


for ep in range(EPISODES):
    # generate a rollout

    # generate an environment
    randomized_environment.sample_env()
    env, env_params = randomized_environment.get_env()


    # reset the environment
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
    total_reward = 0

    # rollout the  whole episode
    while not done:
        obs = current_obs_dict['observation']
        history = episode.get_history()

        noise = agent.action_noise()
        action = agent.evaluate_actor(agent._actor.predict, obs, goal, history) + noise

        new_obs_dict, step_reward, done, info = env.step(action[0])

        new_obs = new_obs_dict['observation']
        achieved = new_obs_dict['achieved_goal']

        episode.add_step(action[0], new_obs, step_reward, achieved, terminal = done)

        total_reward += step_reward

        current_obs_dict = new_obs_dict

    # store the episode in the replay buffer
    replay_buffer.add(episode)

    # replay the episode with HER with probability k
    if random.random() < K:
        new_goal = current_obs_dict['achieved_goal']
        replay_episode = Episode(new_goal, env_params, MAX_STEPS)
        for action, state, achieved_goal, done in zip(episode.get_actions(), episode.get_states(), episode.get_achieved_goals(), episode.get_terminal()):
            # compute the new reward
            step_reward = env.compute_reward(achieved_goal, new_goal, 0)

            # add the fake transition
            replay_episode.add_step(action, state, step_reward, achieved_goal, terminal = done)

        replay_buffer.add(replay_episode)

    # close the environment
    randomized_environment.close_env()

    # perform a batch update of the network if we can sample a big enough batch
    # from the replay buffer

    if replay_buffer.size() > BATCH_SIZE:
        episodes = replay_buffer.sample_batch(BATCH_SIZE)

        s_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_state()])
        a_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_action()])

        next_s_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_state()])

        r_batch = np.zeros([BATCH_SIZE*MAX_STEPS])

        env_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_env()])
        goal_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_goal()])

        history_batch = np.zeros([BATCH_SIZE*MAX_STEPS, MAX_STEPS, agent.get_dim_action()+agent.get_dim_state()])

        t_batch = []

        for i in range(BATCH_SIZE):
            s_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_states())[:-1]
            a_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_actions())[1:]
            next_s_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_states())[1:]
            r_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_rewards())[1:]

            env_batch[i*MAX_STEPS:(i+1)*MAX_STEPS]=np.array(MAX_STEPS*[episodes[i].get_env()])
            goal_batch[i*MAX_STEPS:(i+1)*MAX_STEPS]=np.array(MAX_STEPS*[episodes[i].get_goal()])
            history_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array([episodes[i].get_history(t = t) for t in range(1, MAX_STEPS+1)])


            # WARNING FIXME: needs padding
            t_batch += episodes[i].get_terminal()[1:]

        target_action_batch = agent.evaluate_actor_batch(agent._actor.predict_target, next_s_batch, goal_batch, history_batch)

        predicted_actions = agent.evaluate_actor_batch(agent._actor.predict, next_s_batch, goal_batch, history_batch)

        target_q = agent.evaluate_critic_batch(agent._critic.predict_target, next_s_batch, predicted_actions, goal_batch, history_batch, env_batch)


        y_i = []
        for k in range(BATCH_SIZE*MAX_STEPS):
            if t_batch[k]:
                y_i.append(r_batch[k])
            else:
                y_i.append(r_batch[k] + GAMMA * target_q[k])

        predicted_q_value, _ = agent.train_critic(s_batch, a_batch, goal_batch, history_batch, env_batch, np.reshape(y_i, (BATCH_SIZE*MAX_STEPS, 1)))

        # Update the actor policy using the sampled gradient
        a_outs = agent.evaluate_actor_batch(agent._actor.predict, s_batch, goal_batch, history_batch)                
        grads = agent.action_gradients_critic(s_batch, a_outs, goal_batch, history_batch, env_batch)
        agent.train_actor(s_batch, goal_batch, history_batch, grads[0])

        # Update target networks
        agent.update_target_actor()
        agent.update_target_critic()


    # perform policy evaluation
    if ep % TESTING_INTERVAL == 0:
        success_number = 0
        
        for test_ep in range(TESTING_ROLLOUTS):
            randomized_environment.sample_env()
            env, env_params = randomized_environment.get_env()

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

                action = agent.evaluate_actor(agent._actor.predict_target, obs, goal, history)

                new_obs_dict, step_reward, done, info = env.step(action[0])

                new_obs = new_obs_dict['observation']
                achieved = new_obs_dict['achieved_goal']

                episode.add_step(action[0], new_obs, step_reward, achieved, terminal=done)

                total_reward += step_reward

                current_obs_dict = new_obs_dict

            if info['is_success'] > 0.0:
                success_number += 1

            randomized_environment.close_env()

        print("Testing at episode {}, success rate : {}".format(ep, success_number/TESTING_ROLLOUTS))
