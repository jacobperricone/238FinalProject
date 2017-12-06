import numpy as np
import tensorflow as tf
import gym
from utils import *
from Networks.Baselines import *
import time
import os
import logging
import random

class TRPO():
    def __init__(self, args, env):
        self.observation_space = env.observation_space
        self.action_space =env.action_space
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
        self.hidden_size = 64

        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.oldaction_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
        self.oldaction_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])

        with tf.variable_scope("policy"):
            h1 = fully_connected(self.obs, self.observation_size, self.hidden_size, weight_init, bias_init, "policy_h1")
            h1 = tf.nn.relu(h1)
            h2 = fully_connected(h1, self.hidden_size, self.hidden_size, weight_init, bias_init, "policy_h2")
            h2 = tf.nn.relu(h2)
            h3 = fully_connected(h2, self.hidden_size, self.action_size, weight_init, bias_init, "policy_h3")
            action_dist_logstd_param = tf.Variable((.01 * np.random.randn(1, self.action_size)).astype(np.float32),
                                                   name="policy_logstd")
        # means for each action
        self.action_dist_mu = h3
        # log standard deviations for each actions
        self.action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(self.action_dist_mu)[0], 1)))
        batch_size = tf.shape(self.obs)[0]
        # what are the probabilities of taking self.action, given new and old distributions
        log_p_n = gauss_log_prob(self.action_dist_mu, self.action_dist_logstd, self.action)
        log_oldp_n = gauss_log_prob(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action)

        # tf.exp(log_p_n) / tf.exp(log_oldp_n)
        ratio = tf.exp(log_p_n - log_oldp_n)
        # importance sampling of surrogate loss (L in paper)
        surr = -tf.reduce_mean(ratio * self.advantage)
        var_list = tf.trainable_variables()

        eps = 1e-8
        batch_size_float = tf.cast(batch_size, tf.float32)
        # kl divergence and shannon entropy
        kl = gauss_KL(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action_dist_mu,
                      self.action_dist_logstd) / batch_size_float
        ent = gauss_ent(self.action_dist_mu, self.action_dist_logstd) / batch_size_float

        self.losses = [surr, kl, ent]
        # policy gradient
        self.pg = flatgrad(surr, var_list)

        # KL divergence w/ itself, with first argument kept constant.
        kl_firstfixed = gauss_selfKL_firstfixed(self.action_dist_mu, self.action_dist_logstd) / batch_size_float
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
        # self.vf = VF(self.session)

        self.vf = LinearVF(self.ordering)
        self.get_policy = GetPolicyWeights(self.session, var_list)

    def init_work(self):
        if self.args.monitor:
            self.env.monitor.start('monitor/', force=True)

        self.set_policy = SetPolicyWeights(self.session, tf.trainable_variables())
        self.average_timesteps_in_episode = 1000

    def set_policy_weights(self, parameters):
        self.set_policy(parameters)


    def act(self, obs):
        obs = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd = self.session.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs: obs})

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
                obs = np.expand_dims(obs, 0)
                path = list(map(lambda x: np.concatenate(x), [obs, action_dists_mu, action_dists_logstd]))
                rewards = np.array(rewards)
                returns = discount(rewards, self.args.gamma)
                advantage = np.array(rewards) - self.vf.predict(path[0],len(rewards))
                path.extend([rewards, np.array(actions), returns, advantage])
                # print("obs.shape = {}".format(np.concatenate(obs).shape))
                # print("action_dists_mu.shape = {}".format(np.concatenate(action_dists_mu).shape))
                # print("action_dists_logstd.shape = {}".format(np.concatenate(action_dists_logstd).shape))
                # print("actions.shape = {}".format(np.array(actions).shape))
                # print("rewards.shape = {}".format(rewards.shape))
                # print("returns.shape = {}".format(returns.shape))
                # print("advantage.shape = {}".format(advantage.shape))
                # print("len(path) = {}".format(len(path)))
                return path

    def rollout(self, num_timesteps):
        paths = []
        steps = 0
        episode = 0
        while steps < num_timesteps:
            # print("Running an episode after completing {} timesteps".format(steps))
            # sys.stdout.flush()
            paths.append(self.episode())
            steps += len(paths[episode][self.ordering["rewards"]])
            episode += 1

        self.average_timesteps_in_episode = sum([len(path[self.ordering["rewards"]]) for path in paths]) / len(paths)
        return paths

    def learn(self, paths):
        # puts all the experiences in a matrix: total_timesteps x options
        action_dist_mu = np.concatenate([path[self.ordering["action_dists_mu"]] for path in paths])
        action_dist_logstd = np.concatenate([path[self.ordering["action_dists_logstd"]] for path in paths])
        obs_n = np.concatenate([path[self.ordering["obs"]] for path in paths])
        action_n = np.concatenate([path[self.ordering["actions"]] for path in paths])

        # standardize to mean 0 stddev 1
        advant_n = np.concatenate([path[self.ordering["advantage"]] for path in paths])
        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + 1e-8)

        # train value function / baseline on rollout paths
        self.vf.fit(paths)

        feed_dict = {self.obs: obs_n, self.action: action_n, self.advantage: advant_n,
                     self.oldaction_dist_mu: action_dist_mu, self.oldaction_dist_logstd: action_dist_logstd}

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
        episoderewards = np.array([path[self.ordering["rewards"]].sum() for path in paths])
        stats = {}
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["Entropy"] = entropy_after
        stats["Max KL"] = self.args.max_kl
        stats["Timesteps"] = sum([len(path[self.ordering["rewards"]]) for path in paths])
        stats["KL between old and new distribution"] = kl_after
        stats["Surrogate loss"] = surrogate_after
        return self.get_policy(), stats

    def get_starting_weights(self):
        return self.get_policy()

    def adjust_kl(self, kl_new):
        self.args.max_kl = kl_new
        return
