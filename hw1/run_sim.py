# -*- coding: utf-8 -*-
import tqdm
import numpy as np
import gym
import model_utils
from gym import wrappers


def run_sim(
        policy_fn, envname, num_rollouts, render, capture_dir=None,
        max_timesteps=None):
    """Returns the [[(s,a)..]..] of rollouts."""
    env = gym.make(envname)
    if capture_dir:
        env = wrappers.Monitor(env=env, directory=capture_dir)
    max_steps = max_timesteps or env.spec.timestep_limit
    returns = []
    observations = []
    actions = []
    ro_iter = tqdm.tqdm(
        range(num_rollouts),
        desc='Running sim for rollout',
    )
    for i in ro_iter:
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render or capture_dir:
                env.render()
            if steps >= max_steps:
                break
        returns.append(totalr)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    return {
        'observations': np.array(observations),
        'actions': np.squeeze(np.array(actions)),
        'returns': {
            'list': returns,
            'mean': np.mean(returns),
            'std': np.std(returns),
        },
    }


def sim_to_rollout(**sim_kwargs):
    rollout_data = run_sim(**sim_kwargs)
    return model_utils.Rollout(
        obs=rollout_data['observations'],
        axns=rollout_data['actions'],
    )
