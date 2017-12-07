import numpy as np
import tensorflow as tf
import gym
from utils import *
from Networks.Baselines import *
from Networks.ContinuousPolicy import NetworkContinous
# from Networks.ContinuousPolicy import NetworkDiscrete
import time
import os
import logging
import random

logging.getLogger().setLevel(logging.INFO)

CHECKPOINT_DIR = os.path.join(os.getcwd(), 'Checkpoints')
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

class TRPO():
    def __init__(self, args, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env
        self.args = args
        self.init_net()
        self.init_work()

    def init_net(self):
        self.observation_shape = self.observation_space.shape
        self.observation_size = self.observation_shape[0]

        self.action_size = int(np.prod(self.action_space.shape))
        # self.action_size = self.action_space.n

        keys = ['action_dists_mu', 'action_dists_logstd', 'actions', 'returns', 'advantage']
        self.col_orderings = {'obs': list(range(self.observation_size))}
        self.col_orderings['features'] = list(range(self.observation_size * 2 + 3))
        self.col_orderings = dict(self.col_orderings,
                                  **{keys[i]: [2 * self.observation_size + 3 + i] for i in range(len(keys))})
        self.paths = []
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        # Initialize Policy Network
        self.session = tf.Session(config=config)

        self.net = NetworkContinous("continuous_policy", self.env)
        # self.net = NetworkDiscrete("continuous_policy", self.observation_size, self.action_size)

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
        self.saver = tf.train.Saver(var_list)
        # value function
        self.vf = LinearVF()
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

    def set_policy_weights(self, parameters):
        self.set_policy(parameters)

    def act(self, obs):
        obs = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd = self.session.run([self.net.action_dist_mu, self.net.action_dist_logstd],
                                                              feed_dict={self.net.obs: obs})
        act = action_dist_mu + np.exp(action_dist_logstd) * np.random.randn(*action_dist_logstd.shape)
        return act.ravel(), action_dist_mu, action_dist_logstd

    def episode(self, num_timesteps=0):
        if num_timesteps == 0:
            num_timesteps = self.args.max_pathlength-1
        obs, actions, rewards, action_dists_mu, action_dists_logstd = [], [], [], [], []
        ob = list(filter(self.env.reset()))
        for i in range(self.args.max_pathlength - 1):
            obs.append(ob)
            action, action_dist_mu, action_dist_logstd = self.act(ob)

            # print("action_size = {}".format(self.action_size))
            # print("action.shape = {}".format(action.shape))
            # print("action_dist_mu.shape = {}".format(action_dist_mu.shape))
            # print("action_dist_logstd.shape = {}".format(action_dist_logstd.shape))

            actions.append(action)
            action_dists_mu.append(action_dist_mu)
            action_dists_logstd.append(action_dist_logstd)
            res = self.env.step(int(action))
            ob = list(filter(res[0]))
            rewards.append((res[1]))
            if res[2] or i == self.args.max_pathlength - 2 or i == num_timesteps - 1:
                obs = np.concatenate(np.expand_dims(obs, 0))
                action_dists_mu = np.concatenate(action_dists_mu)
                action_dists_logstd = np.concatenate(action_dists_logstd)
                actions = np.array(actions)
                rewards = np.array(rewards)
                returns = discount(rewards, self.args.gamma)
                rewards, returns = map(lambda x: np.expand_dims(x, -1), [rewards, returns])
                range_array = np.arange(obs.shape[0]).reshape(-1, 1) / 100.0
                ones_array = np.ones((obs.shape[0], 1))
                features = np.concatenate([obs, obs ** 2, range_array, range_array ** 2, ones_array], axis=1)
                advantage = np.expand_dims(rewards.ravel() - self.vf.predict(features), -1)

                # logging.debug("In Episode: obs shape {}".format(obs.shape))
                # logging.debug("In Episode: feat shape {}".format(features.shape))
                # logging.debug("In Episode: action_dists_mu shape {}".format(action_dists_mu.shape))
                # logging.debug("In Episode: action_dists_logstd {}".format(action_dists_logstd.shape))
                # logging.debug("In Episode: actions {}".format(actions.shape))
                # logging.debug("In Episode: returns {}".format(returns.shape))
                # logging.debug("In Episode: advantage {}".format(advantage.shape))

                path = np.hstack((features, action_dists_mu, action_dists_logstd, actions, returns, advantage))
                return path, rewards.sum()

    def rollout(self, num_timesteps):
        # for equal timestep rollouts
        paths = []
        episodes_rewards = np.zeros(2, dtype=np.int)
        steps = 0
        while steps < num_timesteps:
            path, reward = self.episode(num_timesteps - steps)
            steps += path.shape[0]
            paths.append(path)
            if (steps < num_timesteps):  # only record full episodes for averaging!
                episodes_rewards[0] += 1
                episodes_rewards[1] += reward
        paths = np.concatenate(paths, 0)
        return paths, episodes_rewards

    # def rollout(self, num_timesteps):
    #     # for equal number of episode rollouts
    #     paths = []
    #     steps_episodes_rewards = np.zeros(3, dtype=np.int)
    #     while steps_episodes_rewards[0] < num_timesteps:
    #         path, reward = self.episode()
    #         steps_episodes_rewards[0] += path.shape[0]
    #         paths.append(path)
    #         steps_episodes_rewards[1] += 1
    #         steps_episodes_rewards[2] += reward
    #     paths = np.concatenate(paths, 0)
    #     return paths, steps_episodes_rewards

    def learn(self, paths, episodes_rewards):
        obs_n = paths[:, self.col_orderings['obs']]
        action_n = paths[:, self.col_orderings['actions']]
        action_dist_mu = paths[:, self.col_orderings['action_dists_mu']]
        action_dist_logstd = paths[:, self.col_orderings['action_dists_logstd']]
        advant_n = paths[:, self.col_orderings['advantage']].ravel()
        features = paths[:, self.col_orderings['features']]
        returns = paths[:, self.col_orderings['returns']]

        # logging.debug("In Learn: obs_n.shape = {}".format(obs_n.shape))
        # logging.debug("In Learn: action_dist_mu.shape = {}".format(action_dist_mu.shape))
        # logging.debug("In Learn: action_dist_logstd.shape = {}".format(action_dist_logstd.shape))
        # logging.debug("In Learn: action_n.shape = {}".format(action_n.shape))
        # logging.debug("In Learn: advant_n.shape = {}".format(advant_n.shape))
        # logging.debug("In Learn: features.shape = {}".format(features.shape))
        # logging.debug("In Learn: returns.shape = {}".format(returns.shape))

        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + 1e-8)

        # train value function / baseline on rollout paths
        self.vf.fit(features, returns)

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

        def loss2(th):
            self.sff(th)
            # surrogate loss: policy gradient loss
            return self.session.run(self.losses, feed_dict)

        # finds best parameter by starting with a big step and working backwards
        theta = linesearch(loss, thprev, fullstep, negative_g_dot_steppdir / lm)
        # i guess we just take a fullstep no matter what
        theta = thprev + fullstep
        self.sff(theta)

        surrogate_after, kl_after, entropy_after = self.session.run(self.losses, feed_dict)

        # mean rewards per full episode in this iteration
        if episodes_rewards[0] == 0:
            episoderewards = 0
        else:
            episoderewards = episodes_rewards[1] / episodes_rewards[0]
        stats = {}
        stats["Avg_Reward"] = episoderewards
        stats["Entropy"] = entropy_after
        stats["Max KL"] = self.args.max_kl
        stats["Timesteps"] = paths.shape[0]
        stats["Delta_KL"] = kl_after
        stats["Surrogate loss"] = surrogate_after
        stats["Episodes"] = int(episodes_rewards[0])
        return self.get_policy(), stats

    def save_weights(self, checkpoint_name):
        try:
            save_path = self.saver.save(self.session, os.path.join(CHECKPOINT_DIR, checkpoint_name))
            logging.info("Saved model to {}".format(save_path))
        except Exception as e:
            logging.error("Unable to save checkpoint {}".format(e))

    def get_starting_weights(self):
        return self.get_policy()

    def adjust_kl(self, kl_new):
        self.args.max_kl = kl_new
        return