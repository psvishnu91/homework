# -*- coding: utf-8 -*-
"""Solution for HW1 Section3 part 2.

BC performance as a function of num_datapoints used for training.
"""
import collections
import json

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import model_utils
import bclone
import run_sim

plt.style.use('fivethirtyeight')

ROLLOUTS_FILE = 'humanoid_rollouts_50.pkl'
RO_DATASIZES = np.array([10, 20, 30, 40, 50]) * 1000
EPOCHS_LIST = np.array([5000, 4000, 3000, 2000, 1000])
ENVNAME = 'Humanoid-v1'
NUM_ROLLOUTS_SIM = 500
RETURNS_OUTFILE = 'returns_vs_ds.json'


def main():
    print 'Loading rollouts..'
    rollouts = model_utils.load_rollouts(fl=ROLLOUTS_FILE)
    sz_return_map = collections.defaultdict(list)
    for (sz, rolls), epochs in tqdm.tqdm(
            zip(
                iter_rollouts(
                    rollouts=rollouts,
                    ro_datasizes=RO_DATASIZES,
                ),
                EPOCHS_LIST,
            ),
            total=len(RO_DATASIZES),
    ):
        print '\n\nTraining for roll sz: `{:,}`.'.format(len(rolls))
        policy_fn = model_utils.model_to_policy(
            model=bclone.train(
                rolls=rolls,
                epochs=epochs,
                model_func=bclone.MODEL_FUNC,
                to_plot=False,
            ),
        )
        print 'Running simulations for this model'
        returns = run_sim.run_sim(
            policy_fn=policy_fn,
            envname=ENVNAME,
            num_rollouts=NUM_ROLLOUTS_SIM,
            render=False,
            max_timesteps=None,
        )['returns']
        sz_return_map['sz'].append(sz)
        sz_return_map['mean'].append(returns['mean'])
        sz_return_map['std'].append(returns['std'])
    with open(RETURNS_OUTFILE, 'w') as outfile:
        outfile.write(json.dumps(sz_return_map))
    plt.figure(figsize=(15, 15))
    plt.errorbar(
        x=sz_return_map['sz'],
        y=sz_return_map['mean'],
        yerr=sz_return_map['std'],
    )
    plt.title(
        'Mean returns vs training data size (with one stddev '
        'error bars)',
    )
    plt.xlabel('Dataset size')
    plt.ylabel('Mean returns')
    plt.show()


def iter_rollouts(rollouts, ro_datasizes):
    for sz in ro_datasizes:
        yield (
            sz,
            model_utils.Rollout(
                obs=rollouts.obs[:sz],
                axns=rollouts.axns[:sz],
            ),
        )


if __name__ == '__main__':
    main()

