import numpy as np
import tensorflow as tf
import gym
from utils import *
from Networks.Baselines import *
from Networks.ContinuousPolicy import NetworkContinous
import time
import os
import logging
import random

class TRPO():
    def __init__(self, args, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env
        self.args = args
        self.ordering = {"obs": 0, "action_dists_mu": 1, "action_dists_logstd": 2, "rewards": 3, "actions": 4,
                         "returns": 5, "advantage": 6}
        self.init_net()
        self.init_work()

    def init_net(self):
        # previously this was all part of makeModel(self) . . .
        self.observation_size = self.observation_space.shape[0]
        self.action_size = int(np.prod(self.action_space.shape))
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        # Initialize Policy Network
        self.session = tf.Session(config=config)
        self.net = NetworkContinous("continuous_policy", self.observation_size, self.action_size)

        # Calulate Surrogate Loss
        surr = self.calculate_surrogate_loss()
        kl, ent = self.calculate_KL_and_entropy()
        self.losses = [surr, kl, ent]

        batch_size_float = tf.cast(self.net.batch_size, tf.float32)
        var_list = self.net.var_list

        # policy gradient
        self.pg = flatgrad(surr, var_list)

        # KL divergence w/ itself, with first argument kept constant.
        kl_firstfixed = gauss_selfKL_firstfixed(self.net.action_dist_mu, self.net.action_dist_logstd) / batch_size_float
        # gradient of KL w/ itself
        grads = tf.gradients(kl_firstfixed, var_list)
        # what vector we're multiplying by
        self.flat_tangent = tf.placeholder(tf.float32, [None])
        shapes = list(map(var_shape, var_list))
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size

        # gradient of KL w/ itself * tangent
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        # 2nd gradient of KL w/ itself * tangent
        self.fvp = flatgrad(gvp, var_list)
        # the actual parameter values
        self.gf = GetFlat(self.session, var_list)
        # call this to set parameter values
        self.sff = SetFromFlat(self.session, var_list)
        self.session.run(tf.global_variables_initializer())
        # value function
        self.vf = LinearVF(self.ordering)
        self.get_policy = GetPolicyWeights(self.session, var_list)

    def calculate_surrogate_loss(self):
        # what are the probabilities of taking self.action, given new and old distributions
        log_p_n = gauss_log_prob(self.net.action_dist_mu, self.net.action_dist_logstd, self.net.action)
        log_oldp_n = gauss_log_prob(self.net.oldaction_dist_mu, self.net.oldaction_dist_logstd, self.net.action)
        # tf.exp(log_p_n) / tf.exp(log_oldp_n)
        ratio = tf.exp(log_p_n - log_oldp_n)
        # importance sampling of surrogate loss (L in paper)
        surr = -tf.reduce_mean(ratio * self.net.advantage)
        return surr

    def calculate_KL_and_entropy(self):
        eps = 1e-8
        batch_size_float = tf.cast(self.net.batch_size, tf.float32)
        # kl divergence and shannon entropy
        kl = gauss_KL(self.net.oldaction_dist_mu, self.net.oldaction_dist_logstd, self.net.action_dist_mu,
                      self.net.action_dist_logstd) / batch_size_float
        ent = gauss_ent(self.net.action_dist_mu, self.net.action_dist_logstd) / batch_size_float
        return kl, ent

    def init_work(self):
        if self.args.monitor:
            self.env.monitor.start('monitor/', force=True)
        self.set_policy = SetPolicyWeights(self.session, self.net.var_list)
        self.average_timesteps_in_episode = 1000

    def set_policy_weights(self, parameters):
        self.set_policy(parameters)

    def act(self, obs):
        obs = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd = self.session.run([self.net.action_dist_mu, self.net.action_dist_logstd],
                                                              feed_dict={self.net.obs: obs})
        act = action_dist_mu + np.exp(action_dist_logstd) * np.random.randn(*action_dist_logstd.shape)
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
                # obs = np.expand_dims(obs, 0)
                # path = list(map(lambda x: np.concatenate(x), [obs, action_dists_mu, action_dists_logstd]))
                # rewards = np.array(rewards)
                # returns = discount(rewards, self.args.gamma)
                # advantage = np.array(rewards) - self.vf.predict(path[0],len(rewards))
                # path.extend([rewards, np.array(actions), returns, advantage])

                obs = np.expand_dims(obs, 0)
                path = np.concatenate(list(map(lambda x: np.concatenate(x), [obs, action_dists_mu, action_dists_logstd])), axis=1)
                rewards = np.array(rewards)
                returns = discount(rewards, self.args.gamma)
                advantage = np.array(rewards) - self.vf.predict(path[:,:len(ob)],len(rewards))
                rewards = np.expand_dims(rewards, 1)
                returns = np.expand_dims(returns, 1)
                advantage = np.expand_dims(advantage, 1)
                path = np.concatenate([path,rewards,np.array(actions),returns,advantage], axis=1)
                # print("path.shape = {}".format(path.shape))
                return path

    def rollout(self, num_timesteps):
        # paths = []
        paths = self.episode()
        steps = paths.shape[0]
        while steps < num_timesteps:
            # print("Running an episode after completing {} timesteps".format(steps))
            # sys.stdout.flush()
            # paths.append(self.episode())
            paths = np.concatenate([paths,self.episode()], axis = 0)
            steps = paths.shape[0]
            # print("steps = {}".format(steps))

        # self.average_timesteps_in_episode = sum([len(path[self.ordering["rewards"]]) for path in paths]) / len(paths)
        return paths

    def learn(self, paths):
        # action_dist_mu = np.concatenate([path[self.ordering["action_dists_mu"]] for path in paths])
        # action_dist_logstd = np.concatenate([path[self.ordering["action_dists_logstd"]] for path in paths])
        # obs_n = np.concatenate([path[self.ordering["obs"]] for path in paths])
        # action_n = np.concatenate([path[self.ordering["actions"]] for path in paths])
        obs_n = paths[:,self.ordering["obs"]:self.ordering["obs"]+paths.shape[1]-len(self.ordering)+1]
        action_dist_mu = paths[:,self.ordering["action_dists_mu"]+paths.shape[1]-len(self.ordering)]
        action_dist_logstd = paths[:,self.ordering["action_dists_logstd"]+paths.shape[1]-len(self.ordering)]
        action_n = paths[:,self.ordering["actions"]+paths.shape[1]-len(self.ordering)]

        # print("obs_n.shape = {}".format(obs_n.shape))
        # print("action_dist_mu.shape = {}".format(action_dist_mu.shape))
        # print("action_dist_logstd.shape = {}".format(action_dist_logstd.shape))
        # print("action_n.shape = {}".format(action_n.shape))

        # obs_n = np.expand_dims(obs_n, 1)
        action_dist_mu = np.expand_dims(action_dist_mu, 1)
        action_dist_logstd = np.expand_dims(action_dist_logstd, 1)
        action_n = np.expand_dims(action_n, 1)

        # standardize to mean 0 stddev 1
        # advant_n = np.concatenate([path[self.ordering["advantage"]] for path in paths])
        advant_n = paths[:,self.ordering["advantage"]+paths.shape[1]-len(self.ordering)]

        # print("advant_n.shape = {}".format(advant_n.shape))
        advant_n = np.expand_dims(advant_n, 1)

        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + 1e-8)

        # train value function / baseline on rollout paths
        self.vf.fit(paths)

        feed_dict = {self.net.obs: obs_n,
                     self.net.action: action_n,
                     self.net.advantage: advant_n,
                     self.net.oldaction_dist_mu: action_dist_mu,
                     self.net.oldaction_dist_logstd: action_dist_logstd}

        # parameters
        thprev = self.gf()

        # computes fisher vector product: F * [self.pg]
        def fisher_vector_product(p):
            feed_dict[self.flat_tangent] = p
            return self.session.run(self.fvp, feed_dict) + p * self.args.cg_damping

        g = self.session.run(self.pg, feed_dict)

        # solve Ax = g, where A is Fisher information metrix and g is gradient of parameters
        # stepdir = A_inverse * g = x
        stepdir = conjugate_gradient(fisher_vector_product, -g)

        # let stepdir =  change in theta / direction that theta changes in
        # KL divergence approximated by 0.5 x stepdir_transpose * [Fisher Information Matrix] * stepdir
        # where the [Fisher Information Matrix] acts like a metric
        # ([Fisher Information Matrix] * stepdir) is computed using the function,
        # and then stepdir * [above] is computed manually.
        shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))

        lm = np.sqrt(shs / self.args.max_kl)
        # if self.args.max_kl > 0.001:
        #     self.args.max_kl *= self.args.kl_anneal

        fullstep = stepdir / lm
        negative_g_dot_steppdir = -g.dot(stepdir)

        def loss(th):
            self.sff(th)
            # surrogate loss: policy gradient loss
            return self.session.run(self.losses[0], feed_dict)

        # finds best parameter by starting with a big step and working backwards
        theta = linesearch(loss, thprev, fullstep, negative_g_dot_steppdir / lm)
        # i guess we just take a fullstep no matter what
        theta = thprev + fullstep
        self.sff(theta)

        surrogate_after, kl_after, entropy_after = self.session.run(self.losses, feed_dict)

        # episoderewards = np.array([path["rewards"].sum() for path in paths])
        # episoderewards = np.array([path[self.ordering["rewards"]].sum() for path in paths])
        episoderewards = np.sum(paths[:,self.ordering["rewards"]+paths.shape[1]-len(self.ordering)])
        stats = {}
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["Entropy"] = entropy_after
        stats["Max KL"] = self.args.max_kl
        # stats["Timesteps"] = sum([len(path[self.ordering["rewards"]]) for path in paths])
        stats["Timesteps"] = paths.shape[0]
        stats["KL between old and new distribution"] = kl_after
        stats["Surrogate loss"] = surrogate_after
        return self.get_policy(), stats

    def get_starting_weights(self):
        return self.get_policy()

    def adjust_kl(self, kl_new):
        self.args.max_kl = kl_new
        return
