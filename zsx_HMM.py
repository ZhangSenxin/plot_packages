#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


class HMMLearnPred(object):
    def __init__(self, obs, num_state=3, num_obs=8, epsilon=0.01, iter_num=1000, seed_id=42):
        """
        This Class is used to calculate the learning and prediction problems of Hidden Markov Models.
        :param obs: list-like training data
        :param num_state: hidden state counts
        :param num_obs: number of Classification Levels for Observation Data
        :param epsilon: Convergence threshold
        :param iter_num: number of maximum iteration
        :param seed_id: set seed, False is allowed
        """
        self.obs = np.array(obs)
        self.num_state = num_state
        self.num_obs = num_obs
        self.T = len(obs)

        if seed_id:
            np.random.seed(seed_id)

        array_pi = np.random.random(num_state)
        array_pi = array_pi / sum(array_pi)
        self.array_pi = array_pi  # N

        mat_A = np.random.random([num_state, num_state])
        mat_A = (mat_A.T / np.sum(mat_A, axis=1)).T
        self.mat_A = mat_A  # N, N

        mat_B = np.random.random([num_state, num_obs])
        mat_B = (mat_B.T / np.sum(mat_B, axis=1)).T
        self.mat_B = mat_B  # N, N

        self.epsilon = epsilon
        self.iter_num = iter_num

    @staticmethod
    def _forward_function(alpha_t_, mat_a, mat_b, obs_t_next_):
        return np.sum((mat_a.T * alpha_t_).T * mat_b[:, obs_t_next_], axis=0)

    def _forward_alpha(self):
        array_pi = self.array_pi
        mat_A = self.mat_A
        mat_B = self.mat_B

        obs = self.obs
        num_state = self.num_state
        T = self.T

        alpha_1 = np.array([array_pi[i] * mat_B[i, obs[0]] for i in range(num_state)])
        alpha_all = [alpha_1]
        for t in range(1, T):
            alpha_next = self._forward_function(alpha_all[-1], mat_A, mat_B, obs[t])
            alpha_all += [alpha_next]

        alpha_all = np.array(alpha_all)  # T, N

        return alpha_all.T  # N, T

    @staticmethod
    def _backward_function(beta_t_, mat_a, mat_b, obs_t_):
        return np.sum(mat_a * beta_t_ * mat_b[:, obs_t_], axis=1)

    def _backward_beta(self):
        mat_A = self.mat_A
        mat_B = self.mat_B
        obs = self.obs
        num_state = self.num_state
        T = self.T

        beta_T = [1] * num_state
        beta_T = np.array(beta_T)
        beta_all = [beta_T]
        for t in range(1, T):
            beta_next = self._backward_function(beta_all[-1], mat_A, mat_B, obs[-t])
            beta_all += [beta_next]

        beta_all = np.array(beta_all[::-1])  # T, N

        return beta_all.T  # N, T

    @staticmethod
    def _gamma_ksi(alpha_all, beta_all, mat_a, mat_b, obs, num_state_, T):
        gamma_up = alpha_all * beta_all  # N, T
        gamma = gamma_up / np.sum(gamma_up, axis=0)  # N, T

        # (N, T-1)' dot (N, N) * (N, T-1)' * (N, T-1)'
        denominator = np.dot(alpha_all[:, :-1].T, mat_a) * \
                      mat_b[:, obs[1:]].T * \
                      beta_all[:, 1:].T
        denominator = np.sum(denominator, axis=1)  # (T-1)
        denominator = np.expand_dims(denominator, 0).repeat(num_state_, axis=0)  # N * (T-1)
        denominator = np.expand_dims(denominator, 0).repeat(num_state_, axis=0)  # N * (N, T-1)

        numerator1 = np.expand_dims(mat_a, 2).repeat(T - 1, axis=2)  # (N, N) * T-1
        numerator2 = np.expand_dims(beta_all[:, 1:], 0).repeat(num_state_, axis=0)  # N * (N, T-1)
        numerator3 = np.expand_dims(mat_b[:, obs[1:]], 0).repeat(num_state_, axis=0)  # N * (N, T-1)
        numerator4 = np.expand_dims(alpha_all[:, :-1], 1).repeat(num_state_, axis=1)  # (N, ) * N * (T-1, )
        numerator = numerator1 * numerator2 * numerator3 * numerator4

        ksi = numerator / denominator
        ksi = ksi.transpose(2, 0, 1)

        return gamma, ksi

    def step_M(self, silence=True):
        """
        EM step to perform parameter estimation.
        :param silence: print loss per step and last parameters or not
        :return: None
        """
        mat_A = self.mat_A
        mat_B = self.mat_B
        obs = self.obs
        num_state = self.num_state
        num_obs = self.num_obs
        T = self.T

        count = 0
        while True:
            count += 1
            alpha_all = self._forward_alpha()
            beta_all = self._backward_beta()

            gamma, ksi = self._gamma_ksi(alpha_all, beta_all, mat_A, mat_B, obs, num_state, T)

            array_pi_new = gamma[:, 0]  # N

            mat_A_new_up = np.sum(ksi, axis=0)  # N_alpha, N_beta
            mat_A_new_down_sum = np.sum(gamma[:, :-1], axis=1)  # N_alpha
            mat_A_new_down = np.array([[val] * num_state for val in mat_A_new_down_sum])  # N_alpha, N_beta
            mat_A_new = mat_A_new_up / mat_A_new_down  # N_alpha, N_beta

            mat_B_new_up = [np.sum(gamma[:, obs == k], axis=1) for k in range(num_obs)]  # K, N
            mat_B_new_up = np.array(mat_B_new_up).T  # N, K
            mat_B_new_down_sum = np.sum(gamma, axis=1)  # N
            mat_B_new_down = np.array([[val] * num_obs for val in mat_B_new_down_sum])  # N, K
            mat_B_new = mat_B_new_up / mat_B_new_down  # N, K

            diff_pi = np.sum(abs(array_pi_new - self.array_pi))
            diff_A = np.sum(abs(mat_A_new - self.mat_A))
            diff_B = np.sum(abs(mat_B_new - self.mat_B))

            self.array_pi = array_pi_new
            self.mat_A = mat_A_new
            self.mat_B = mat_B_new

            if not silence:
                str_pi = exactly_round(diff_pi, 3)
                str_A = exactly_round(diff_A, 3)
                str_B = exactly_round(diff_B, 3)
                print(count, str_pi, str_A, str_B, sep='\t')

            if np.max([diff_pi, diff_A, diff_B]) < self.epsilon or count >= self.iter_num:
                if count >= self.iter_num:
                    print('not Convergence')

                if not silence:
                    print('\npi result')
                    print(array_pi_new)
                    print('\nmat_A result')
                    print(mat_A_new)
                    print('\nmat_B result')
                    print(mat_B_new)
                break

    def predict(self, obs=None, mat_A=None, mat_B=None, array_pi=None, num_state=None):
        """
        Viterbi algorithm to get prediction.
        :param obs: observation data, default 'None' will use data from step_M()
        :param mat_A: State transition matrix
        :param mat_B: emission probability matrix
        :param array_pi: initial state vector
        :param num_state: hidden state counts
        :return: predicte states list, corresponding probability list
        """
        if mat_A is None:
            mat_A = self.mat_A
        if mat_B is None:
            mat_B = self.mat_B
        if array_pi is None:
            array_pi = self.array_pi
        if num_state is None:
            num_state = self.num_state
        if obs is None:
            obs = self.obs
            T = self.T
        else:
            T = len(obs)

        # prob
        delta = np.zeros((T, num_state))  # T, N
        # strategy
        psi = np.zeros((T, num_state))  # T, N

        # viterbi algorithm
        delta[0] = array_pi.reshape((1, num_state)) * mat_B[:, obs[0]].reshape((1, num_state))
        for t in range(1, T):
            for i in range(num_state):
                delta_t_i = delta[t - 1] * mat_A[:, i]
                delta_t_i = delta_t_i * mat_B[i, obs[t]]
                delta[t, i] = np.max(delta_t_i)
                psi[t][i] = np.argmax(delta_t_i)

        delta_norm = (delta.T / np.sum(delta, axis=1)).T
        states_ = [int(np.argmax(delta[t]))]
        probs_ = [np.max(delta_norm[t])]
        for t in range(1, T):
            states_ += [int(psi[-t, states_[-1]])]
            probs_ += [delta_norm[-t, states_[-1]]]
        states_ = states_[::-1]
        probs_ = probs_[::-1]

        return states_, probs_


def exactly_round(float_like_, round_num_):
    """
    Round number to a string form in a given decimal places.
    :param float_like_: float or int object
    :param round_num_: round the given number
    :return: string number
    """
    float_like_ = str(np.round(float(float_like_), round_num_))
    try:
        length_ = len(float_like_.split('.')[1])
    except IndexError:
        print(float_like_, 'error')
        return 'n.d.'

    if length_ == round_num_:
        return float_like_
    else:
        diff = round_num_ - length_

        return float_like_ + '0' * diff


if __name__ == "__main__":
    import time

    obs = np.random.randint(0, 7, 100)

    HLP = HMMLearnPred(obs,
                       num_state=3, num_obs=7,
                       epsilon=0.001, iter_num=1000,
                       seed_id=42)
    t1 = time.time()
    HLP.step_M(silence=False)
    states, probs = HLP.predict()
    t2 = time.time()

    print(states)
    print(probs)

    try:
        import zsx_some_tools as st
        print(st.time_trans(t2 - t1, second_round_num=3))
    except ModuleNotFoundError:
        print(t2 - t1)

