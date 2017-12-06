from utils import *
import numpy as np
import tensorflow as tf

class NetworkContinous(object):
    def __init__(self, scope, obs_size, act_size):
        self.observation_size = obs_size
        self.action_size = act_size
        self.hidden_size = 64
        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)

        with tf.variable_scope("%s_shared" % scope):
            self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
            self.action = tf.placeholder(tf.float32, [None, self.action_size])
            self.advantage = tf.placeholder(tf.float32, [None])
            self.oldaction_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
            self.oldaction_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])

            h1 = fully_connected(self.obs, self.observation_size, self.hidden_size, weight_init, bias_init, "{}_policy_h1".format(scope))
            h1 = tf.nn.relu(h1)
            h2 = fully_connected(h1, self.hidden_size, self.hidden_size, weight_init, bias_init, "{}_policy_h2".format(scope))
            h2 = tf.nn.relu(h2)
            self.action_dist_mu = fully_connected(h2, self.hidden_size, self.action_size, weight_init, bias_init, "{}_policy_h3".format(scope))

            self.batch_size =  tf.shape(self.obs)[0]
            self.action_dist_logstd_param = tf.Variable((.01 * np.random.randn(1, self.action_size)).astype(np.float32),
                                                   name="{}_policy_logstd".format(scope))

            self.action_dist_logstd = tf.tile(self.action_dist_logstd_param, tf.stack((tf.shape(self.action_dist_mu)[0], 1)))

            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]
