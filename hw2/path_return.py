"""Contains functions to compute Q or path returns.

Test
====

Sample input::

    path = {'reward': [1, 1, 1, 1, 1]}
    gamma = 0.9

    print('reward-to-go:', _compute_Q_reward_to_go(path, gamma))
    print('not-reward-to-go:', _compute_Q_same_rewards(path, gamma))
    print('reward-to-go_multi:', compute_Qs([path, path], gamma, reward_to_go=True))
    print('not-reward-to-go_multi:', compute_Qs([path, path], gamma, reward_to_go=False))

Sample output::

    ('reward-to-go:', [4.0951, 3.439, 2.71, 1.9, 1])
    ('not-reward-to-go:', [4.0951, 4.0951, 4.0951, 4.0951, 4.0951])
    ('reward-to-go_multi:', array([4.0951, 3.439 , 2.71  , 1.9   , 1.,
        4.0951, 3.439 , 2.71  , 1.9   , 1.    ]))
    (
        'not-reward-to-go_multi:',
        array([4.0951, 4.0951, 4.0951, 4.0951, 4.0951, 4.0951, 4.0951
             4.0951, 4.0951, 4.0951])
    )
"""
import numpy as np


def pathlength(path):
    return len(path['reward'])


def compute_Qs(paths, gamma, reward_to_go):
    return np.concatenate(
        [
            _compute_Q_single(path=path, gamma=gamma, reward_to_go=reward_to_go)
            for path in paths
        ],
    )


def _compute_Q_single(path, gamma, reward_to_go):
    if reward_to_go:
        return _compute_Q_reward_to_go(path=path, gamma=gamma)
    else:
        return _compute_Q_same_rewards(path=path, gamma=gamma)


def _compute_Q_reward_to_go(path, gamma):
    n = pathlength(path=path)
    returns = [None] * n
    ret_ix = n - 1
    for reward in reversed(path['reward']):
        returns[ret_ix] = reward
        if ret_ix != (n-1):
            returns[ret_ix] += gamma * returns[ret_ix + 1]
        ret_ix -= 1
    return returns


def _compute_Q_same_rewards(path, gamma):
    n = pathlength(path=path)
    return_ = 0
    for t, reward in enumerate(path['reward']):
        return_ += reward * gamma**(t)
    return [return_] * n
