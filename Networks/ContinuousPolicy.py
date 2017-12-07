from utils import *
import numpy as np
import tensorflow as tf
import prettytensor as pt
from utils import *

class NetworkContinous(object):
    def __init__(self, scope, env):
        self.env = env
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = int(np.prod(self.env.action_space.shape))

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


    def act(self, sess, obs):
        action_dist_mu, action_dist_logstd = sess.run([self.action_dist_mu, self.action_dist_logstd],
                                                              feed_dict={self.obs: obs})
        act = action_dist_mu + np.exp(action_dist_logstd) * np.random.randn(*action_dist_logstd.shape)
        return act.ravel(), action_dist_mu, action_dist_logstd


class NetworkDiscrete(object):
    def __init__(self, scope, env):
        self.env = env
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.hidden_size = 32

        with tf.variable_scope("%s_shared" % scope):
            self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
            self.action = tf.placeholder(tf.int64, [None])
            self.advantage = tf.placeholder(tf.float32, [None])
            self.oldaction_dist_n = tf.placeholder(tf.float32, [None, self.action_size], name = "old_action")


            self.action_dist_n, _ = (pt.wrap(self.obs).
                                fully_connected(self.hidden_size, activation_fn=tf.nn.relu).
                                fully_connected(self.hidden_size, activation_fn=tf.nn.relu).
                                softmax_classifier(self.action_size))

            self.batch_size =  tf.shape(self.obs)[0]


            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

    def act(self, sess, obs, train = True):
        action_dist_n = sess.run(self.action_dist_n, {self.obs: obs})

        if train:
            action = int(cat_sample(action_dist_n)[0])
        else:
            action = int(np.argmax(action_dist_n))

        return action, action_dist_n

