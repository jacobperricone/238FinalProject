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

    # def run(self):
    #     self.env = gym.make(self.args.task)
    #     self.env.seed(randint(0,999999))
    #     if self.monitor:
    #         self.env.monitor.start('monitor/', force=True)

    #     # tensorflow variables (same as in model.py)
    #     self.observation_size = self.env.observation_space.shape[0]
    #     self.action_size = int(np.prod(self.env.action_space.shape))
    #     # print(self.action_size)
    #     self.hidden_size = 64
    #     weight_init = tf.random_uniform_initializer(-0.05, 0.05)
    #     bias_init = tf.constant_initializer(0)
    #     # tensorflow model of the policy
    #     self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
    #     self.debug = tf.constant([2,2])
    #     with tf.variable_scope("policy-a"):
    #         h1 = fully_connected(self.obs, self.observation_size, self.hidden_size, weight_init, bias_init, "policy_h1")
    #         h1 = tf.nn.relu(h1)
    #         h2 = fully_connected(h1, self.hidden_size, self.hidden_size, weight_init, bias_init, "policy_h2")
    #         h2 = tf.nn.relu(h2)
    #         h3 = fully_connected(h2, self.hidden_size, self.action_size, weight_init, bias_init, "policy_h3")
    #         action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, self.action_size)).astype(np.float32), name="policy_logstd")
    #     self.action_dist_mu = h3
    #     self.action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(self.action_dist_mu)[0], 1)))

    #     config = tf.ConfigProto(
    #         device_count = {'GPU': 0}
    #     )

    #     print("AT A")

    #     self.session = tf.Session(config=config)

    #     print("AT B")

    #     self.session.run(tf.global_variables_initializer())
    #     var_list = tf.trainable_variables()

    #     self.set_policy = SetPolicyWeights(self.session, var_list)

    #     print("AT C")

    #     while True:
    #         # get a task, or wait until it gets one
    #         next_task = self.task_q.get(block=True)
    #         if next_task == 1:
    #             # the task is an actor request to collect experience
    #             path = self.rollout()
    #             self.task_q.task_done()
    #             self.result_q.put(path)
    #         elif next_task == 2:
    #             print("kill message")
    #             if self.monitor:
    #                 self.env.monitor.close()
    #             self.task_q.task_done()
    #             break
    #         else:
    #             # the task is to set parameters of the actor policy
    #             self.set_policy(next_task)
    #             # super hacky method to make sure when we fill the queue with set parameter tasks,
    #             # an actor doesn't finish updating before the other actors can accept their own tasks.
    #             time.sleep(0.1) ## Don't think we need this
    #             self.task_q.task_done()
    #     return
