import tensorflow as tf
import tflearn

UNITS = 128
MAX_STEPS = 100

class Critic:
    def __init__(self, session, dim_state, dim_goal, dim_action, dim_env, env, tau, learning_rate, num_actor_vars):
        self._sess = session

        self._dim_state = dim_state
        self._dim_action = dim_action
        self._dim_env = dim_env
        self._dim_goal = dim_goal
        self._action_bound = env.action_space.high

        self._learning_rate = learning_rate
        self._tau = tau


        self._net_inputs, self._net_out = self.create_network()

        self._net_input_env, self._net_input_goal, self._net_input_action, self._net_input_state, self._net_input_history = self._net_inputs

        self._network_params = tf.trainable_variables()[num_actor_vars:]

        self._target_inputs, self._target_out = self.create_network()

        self._target_input_env, self._target_input_goal, self._target_input_action, self._target_input_state, self._target_input_history = self._target_inputs

        self._target_network_params = tf.trainable_variables()[(len(self._network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self._update_target_network_params = \
            [self._target_network_params[i].assign(tf.multiply(self._network_params[i], self._tau) \
            + tf.multiply(self._target_network_params[i], 1. - self._tau))
                for i in range(len(self._target_network_params))]

        # Network target (y_i)
        self._predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self._loss = tflearn.mean_square(self._predicted_q_value, self._net_out)
        self._optimize = tf.train.AdamOptimizer(
            self._learning_rate).minimize(self._loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self._action_grads = tf.gradients(self._net_out, self._net_input_action)    

    def create_network(self):
        input_state = tflearn.input_data(shape=[None, self._dim_state])
        input_goal = tflearn.input_data(shape=[None, self._dim_goal])
        input_action = tflearn.input_data(shape=[None, self._dim_action])
        input_env = tflearn.input_data(shape=[None, self._dim_env])

        input_history = tflearn.input_data(shape=[None, MAX_STEPS, self._dim_action + self._dim_state])

        input_ff = tflearn.merge(
            [input_env, input_goal, input_action, input_state], 'concat')

        ff_branch = tflearn.fully_connected(input_ff, UNITS)
        ff_branch = tflearn.activations.relu(ff_branch)

        #recurrent_branch = tflearn.fully_connected(inputs, UNITS)
        #recurrent_branch = tflearn.activations.relu(recurrent_branch)
        recurrent_branch = tflearn.lstm(input_history, UNITS, dynamic=True)

        merged_branch = tflearn.merge([ff_branch, recurrent_branch], 'concat')
        merged_branch = tflearn.fully_connected(merged_branch, UNITS)
        merged_branch = tflearn.activations.relu(merged_branch)

        merged_branch = tflearn.fully_connected(merged_branch, UNITS)
        merged_branch = tflearn.activations.relu(merged_branch)

        weights_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            merged_branch, 1, activation='linear', weights_init=weights_init)

        return [input_env, input_goal, input_action, input_state, input_history], out


    def train(self, input_env, input_state, input_goal, input_action, input_history, predicted_q_value):
        return self._sess.run([self._net_out, self._optimize], feed_dict={
            self._net_input_env: input_env,
            self._net_input_state:  input_state,
            self._net_input_goal:  input_goal,
            self._net_input_action:  input_action,
            self._net_input_history:  input_history,

            self._predicted_q_value: predicted_q_value
        })

    def predict(self, input_env, input_state, input_goal, input_action, input_history):
        return self._sess.run(self._net_out, feed_dict={
            self._net_input_env: input_env,
            self._net_input_state: input_state,
            self._net_input_goal: input_goal,
            self._net_input_action: input_action,
            self._net_input_history: input_history,
        })

    def predict_target(self, input_env, input_state, input_goal, input_action, input_history):
        return self._sess.run(self._target_out, feed_dict={
            self._target_input_env: input_env,
            self._target_input_state: input_state,
            self._target_input_goal: input_goal,
            self._target_input_action: input_action,
            self._target_input_history: input_history,
        })

    def action_gradients(self, input_env, input_state, input_goal, input_action, input_history):
        return self._sess.run(self._action_grads, feed_dict={
            self._net_input_env: input_env,
            self._net_input_state: input_state,
            self._net_input_goal: input_goal,
            self._net_input_action: input_action,
            self._net_input_history: input_history
        })

    def update_target_network(self):
        self._sess.run(self._update_target_network_params)
