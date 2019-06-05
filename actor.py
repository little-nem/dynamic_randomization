import tensorflow as tf
import tflearn

UNITS = 128
MAX_STEPS = 50

class Actor:
    def __init__(self, session, dim_state, dim_goal, dim_action, env, tau, learning_rate, batch_size):
        self._sess = session
        
        self._dim_state = dim_state
        self._dim_action = dim_action
        self._dim_goal = dim_goal
        self._action_bound = env.action_space.high
        self._internal_memory = []
        self._tau = tau
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        self._net_inputs, self._net_out, self._net_scaled_out = self.create_network()
        self._net_input_state, self._net_input_goal, self._net_input_history = self._net_inputs
        self._network_params = tf.trainable_variables()

        self._target_inputs, self._target_out, self._target_scaled_out = self.create_network()
        self._target_input_state, self._target_input_goal, self._target_input_history = self._target_inputs

        self._target_network_params = tf.trainable_variables()[len(self._network_params):]

        # op to initialize the target network with the same values as the online network
        self._initialize_target_network_params = [self._target_network_params[i].assign(self._network_params[i]) for i in range(len(self._target_network_params))]

        # op for periodically updating target network with online network weights
        self._update_target_network_params = [self._target_network_params[i].assign(tf.multiply(self._network_params[i], self._tau) + tf.multiply(self._target_network_params[i], 1. - self._tau)) for i in range(len(self._target_network_params))]

        # This gradient will be provided by the critic network
        self._action_gradient = tf.placeholder(tf.float32, [None, self._dim_action])

        # Combine the gradients here
        self._unnormalized_actor_gradients = tf.gradients(self._net_scaled_out, self._network_params, -self._action_gradient)
        self._actor_gradients = list(map(lambda x: tf.div(x, self._batch_size), self._unnormalized_actor_gradients))

        # Optimization Op
        self._optimize = tf.train.AdamOptimizer(self._learning_rate).apply_gradients(zip(self._actor_gradients, self._network_params))

        self._num_trainable_vars = len(self._network_params) + len(self._target_network_params)

    def create_network(self):
        input_state = tflearn.input_data(shape=[None, self._dim_state], name='input_state')
        input_goal = tflearn.input_data(shape=[None, self._dim_goal], name='input_goal')

        input_memory = tflearn.input_data(shape=[None, MAX_STEPS, self._dim_state + self._dim_action])

        input_ff = tflearn.merge([input_goal, input_state], 'concat')

        ff_branch = tflearn.fully_connected(input_ff, UNITS)
        ff_branch = tflearn.activations.relu(ff_branch)

        # recurrent_branch = tflearn.fully_connected(input_memory, UNITS)
        # recurrent_branch = tflearn.activations.relu(recurrent_branch)
        recurrent_branch = tflearn.lstm(input_memory, UNITS, dynamic=True)

        merged_branch = tflearn.merge([ff_branch, recurrent_branch], 'concat')
        merged_branch = tflearn.fully_connected(merged_branch, UNITS)
        merged_branch = tflearn.activations.relu(merged_branch)

        merged_branch = tflearn.fully_connected(merged_branch, UNITS)
        merged_branch = tflearn.activations.relu(merged_branch)

        weights_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            merged_branch, self._dim_action, activation='tanh', weights_init=weights_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self._action_bound)
        return [input_state, input_goal, input_memory], out, scaled_out

    def train(self, input_state, input_goal, input_history, a_gradient):
        self._sess.run(self._optimize, feed_dict={
            self._net_input_state: input_state,
            self._net_input_goal: input_goal,
            self._net_input_history: input_history,
            self._action_gradient: a_gradient
        })

    def predict(self, input_state, input_goal, input_history):
        return self._sess.run(self._net_scaled_out, feed_dict={
            self._net_input_state: input_state,
            self._net_input_goal: input_goal,
            self._net_input_history: input_history,
        })

    def predict_target(self, input_state, input_goal, input_history):
        return self._sess.run(self._target_scaled_out, feed_dict={
            self._target_input_state: input_state,
            self._target_input_goal: input_goal,
            self._target_input_history: input_history,
        })

    def update_target_network(self):
        self._sess.run(self._update_target_network_params)

    def initialize_target_network(self):
        self._sess.run(self._initialize_target_network_params)

    def get_num_trainable_vars(self):
        return self._num_trainable_vars
