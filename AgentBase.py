import numpy as np
import tensorflow as tf
import gym
from utils import *
from rollouts import *
from value_function import *
import time
import os
import logging
import random



class TRPOAgentBase():

	def __init__(self, args, env):
		self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.args = args


        if self.args.monitor:
            self.env.monitor.start('monitor/', force=True)

        self.set_policy = SetPolicyWeights(self.session, tf.trainable_variables())
        self.average_timesteps_in_episode = 1000


    def set_policy_weights(self, parameters):
        self.set_policy(parameters)
