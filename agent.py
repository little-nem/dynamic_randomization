import gym
import tensorflow as tf
import numpy as np

from actor import Actor
from critic import Critic
from noise import OrnsteinUhlenbeckActionNoise


MAX_STEPS = 50
TAU = 5e-3
LEARNING_RATE = 1e-3

class Agent:
    def __init__(self, experiment, batch_size):
        self._dummy_env = gym.make(experiment)
        self._sess = tf.Session()
        
        self._sum_writer = tf.summary.FileWriter('logs/', self._sess.graph)

        # Hardcoded for now
        self._dim_state = 10
        self._dim_goal = 3
        self._dim_action = self._dummy_env.action_space.shape[0]
        self._dim_env = 1
        self._batch_size = batch_size

        # agent noise
        self._action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self._dim_action))

        self._actor = Actor(self._sess,
            self._dim_state, self._dim_goal, self._dim_action, self._dummy_env, TAU, LEARNING_RATE, self._batch_size)

        self._critic = Critic(self._sess,
            self._dim_state, self._dim_goal, self._dim_action, self._dim_env, self._dummy_env, TAU, LEARNING_RATE, self._actor.get_num_trainable_vars(), self._sum_writer)

        self._saver = tf.train.Saver(max_to_keep=None)

        self._sess.run(tf.global_variables_initializer())

        self._actor.initialize_target_network()
        self._critic.initialize_target_network()

        # training monitoring
        self._success_rate = tf.Variable(0., name="success_rate")
        self._python_success_rate = tf.placeholder("float32", [])

        self._update_success_rate = self._success_rate.assign(self._python_success_rate)
        self._merged = tf.summary.scalar("successrate", self._update_success_rate)
        #self._merged = tf.summary.merge(s)


        #writer = tf.summary.FileWriter('logs/')
        #writer.add_summary(
        #writer.add_graph(tf.get_default_graph())
        #writer.flush()
        #
    def get_dim_state(self):
        return self._dim_state

    def get_dim_action(self):
        return self._dim_action
        
    def get_dim_env(self):
        return self._dim_env

    def get_dim_goal(self):
        return self._dim_goal

    def evaluate_actor(self, actor_predict, obs, goal, history):

        assert (history.shape[0] == MAX_STEPS), "history must be of size MAX_STEPS"

        obs = obs.reshape(1, self._dim_state)
        goal = goal.reshape(1, self._dim_goal)
        history = history.reshape(1, history.shape[0], history.shape[1])

        return actor_predict(obs, goal, history)

    def evaluate_actor_batch(self, actor_predict, obs, goal, history):

        return actor_predict(obs, goal, history)

    def evaluate_critic(self, critic_predict, obs, action, goal, history, env):
        obs = obs.reshape(1, self._dim_state)
        goal = goal.reshape(1, self._dim_goal)
        action = action.reshape(1, self._dim_action)
        history = history.reshape(1, history.shape[0], history.shape[1])
        env = env.reshape(1, self._dim_env)

        return critic_predict(env, obs, goal, action, history)

    def evaluate_critic_batch(self, critic_predict, obs, action, goal, history, env):
        return critic_predict(env, obs, goal, action, history)

    def train_critic(self, obs, action, goal, history, env, predicted_q_value):
        return self._critic.train(env, obs, goal, action, history, predicted_q_value)

    def train_actor(self, obs, goal, history, a_gradient):
        return self._actor.train(obs, goal, history, a_gradient)

    def action_gradients_critic(self, obs, action, goal, history, env):
        return self._critic.action_gradients(env, obs, goal, action, history)

    def update_target_actor(self):
        self._actor.update_target_network()

    def update_target_critic(self):
        self._critic.update_target_network()

    def action_noise(self):
        return self._action_noise()

    def update_success(self, success_rate, step):
        _, result = self._sess.run([self._update_success_rate, self._merged], feed_dict={self._python_success_rate: success_rate})
        self._sum_writer.add_summary(result, step)

    def save_model(self, filename):
        self._saver.save(self._sess, filename)

    def load_model(self, filename):
        self._saver.restore(self._sess, filename)
