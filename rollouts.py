import numpy as np
import tensorflow as tf
from utils import *
import gym
import time
import copy
from random import randint

import sys


class Actor():
    def __init__(self, args, monitor, learner, learner_env):
        self.args = args
        self.monitor = monitor

        self.env = learner_env
        self.env.seed(randint(0,999999))
        if self.monitor:
            self.env.monitor.start('monitor/', force=True)

        self.learner = learner
        self.set_policy = SetPolicyWeights(self.learner.session, tf.trainable_variables())

        # # we will start by running (args.timesteps_per_batch / 1000) episodes for the first iteration
        # self.average_timesteps_in_episode = 1000

    def set_policy_weights(self, parameters):
        self.set_policy(parameters)

    def act(self, obs):
        obs = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd = self.learner.session.run([self.learner.action_dist_mu, self.learner.action_dist_logstd], feed_dict={self.learner.obs: obs})
        # samples the guassian distribution
        act = action_dist_mu + np.exp(action_dist_logstd)*np.random.randn(*action_dist_logstd.shape)
        return act.ravel(), action_dist_mu, action_dist_logstd

    def episode(self):
        obs, actions, rewards, action_dists_mu, action_dists_logstd = [], [], [], [], []
        ob = list(filter(self.env.reset()))
        for i in range(self.args.max_pathlength - 1):
            obs.append(ob)
            action, action_dist_mu, action_dist_logstd = self.act(ob)
            actions.append(action)
            action_dists_mu.append(action_dist_mu)
            action_dists_logstd.append(action_dist_logstd)
            res = self.env.step(int(action))
            ob = list(filter(res[0]))
            rewards.append((res[1]))
            if res[2] or i == self.args.max_pathlength - 2:
                path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                        "action_dists_mu": np.concatenate(action_dists_mu),
                        "action_dists_logstd": np.concatenate(action_dists_logstd),
                        "rewards": np.array(rewards),
                        "actions":  np.array(actions)}
                return path


    def rollout(self, num_timesteps):
        paths = []
        steps = 0
        episode = 0
        while steps < num_timesteps:
            # print("Running an episode after completing {} timesteps".format(steps))
            # sys.stdout.flush()
            paths.append(self.episode())
            steps += len(paths[episode]["rewards"])
            episode += 1

        self.average_timesteps_in_episode = sum([len(path["rewards"]) for path in paths]) / len(paths)
        return paths


