import numpy as np
import gym

from environment import RandomizedEnvironment
from agent import Agent
from replay_buffer import ReplayBuffer

EPISODES = 1000

experiment = "FetchReach-v1"
env = gym.make(experiment)

# Initialize networks
BATCH_SIZE = 128
BUFFER_SIZE = 100000
MAX_STEPS = 100
GAMMA = 0.99

agent = Agent(experiment, BATCH_SIZE)
randomized_environment = RandomizedEnvironment(experiment, [], [])

replay_buffer = ReplayBuffer(BUFFER_SIZE)

dim_history_atom = agent._dim_state + agent._dim_action

randomized_environment.sample_env()
env, env_params = randomized_environment.get_env()

success = 0

for episode in range(EPISODES):
    history = np.array(MAX_STEPS*[np.zeros(dim_history_atom)])
    # generate a rollout

    obs_dict = env.reset()
    last_action = env.action_space.sample() # fake last_action, to feed the network

    obs = obs_dict['observation']
    history = np.append(history, [np.concatenate((last_action, obs))], axis = 0)[1:]

    done = False

    print("Episode : {}".format(episode))
    tot_rew = 0

    while not done:
        obs = obs_dict['observation']
        goal = obs_dict['desired_goal']
        action = agent.evaluate_actor(agent._actor.predict, obs, goal, history)

        if(episode > 600):
            env.render()

        new_obs_dict, step_reward, done, info = env.step(action[0])
        tot_rew += step_reward
        new_obs = new_obs_dict['observation']

        history = np.append(history, [np.concatenate((action[0], obs))], axis = 0)[1:]

        replay_buffer.add(obs, action, step_reward, done, new_obs, history, env_params, goal)

        obs = new_obs

        if done and info['is_success'] > 0.01:
            success += 1
            print("Success ! {}/{} ({})".format(success, episode, success/episode))

        if replay_buffer.size() > BATCH_SIZE and done:
            s_batch, a_batch, r_batch, t_batch, s2_batch, history_batch, env_batch, goal_batch = replay_buffer.sample_batch(BATCH_SIZE)

            target_action_batch = agent.evaluate_actor_batch(agent._actor.predict_target, s2_batch, goal_batch, history_batch)

            predicted_actions = agent.evaluate_actor_batch(agent._actor.predict, s2_batch, goal_batch, history_batch)
            target_q = agent.evaluate_critic_batch(agent._critic.predict_target, s2_batch, predicted_actions, goal_batch, history_batch, env_batch)

            y_i = []
            for k in range(BATCH_SIZE):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + GAMMA * target_q[k])

            predicted_q_value, _ = agent.train_critic(s_batch, a_batch, goal_batch, history_batch, env_batch, np.reshape(y_i, (BATCH_SIZE, 1)))


            # Update the actor policy using the sampled gradient
            a_outs = agent.evaluate_actor_batch(agent._actor.predict, s_batch, goal_batch, history_batch)                
            grads = agent.action_gradients_critic(s_batch, a_outs, goal_batch, history_batch, env_batch)
            agent.train_actor(s_batch, goal_batch, history_batch, grads[0])

            # Update target networks
            agent.update_target_actor()
            agent.update_target_critic()
    print("Reward : {}".format(tot_rew))
