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


    def get_action_dist_means_n(self, session, obs):
        return session.run(self.action_dist_means_n,
                         {self.obs: obs})

