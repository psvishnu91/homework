import collections
import multiprocessing as mlp
import random

import numpy as np

import train_pg
import tensorboard_pycrayon as tb


NUM_SETTINGS = 5
NUM_PROCESSES = 4
EXPNAME_SHORT_FORM = 'InvPen'
HYPERPARAM_CHOICES = {
    'discount': [1.],#[0.9, 0.95, 0.99, 1.],
    'learning_rate': [5e-3],#10 ** np.random.uniform(low=-6, high=-1, size=(10)),
    'n_layers': [1, 2, 3],
    'size': [16, 32, 64, 128, 256],
    'batch_size': [500, 1000, 2000, 3000, 4000, 5000],
}
HYPERPARAM_SHORT_FORM = {
    'discount': 'gma',
    'learning_rate': 'lr',
    'n_layers': 'lyr',
    'size': 'unt',
    'batch_size': 'bsz',
}
DEFAULT_PARAMS = {
    'env_name': 'InvertedPendulum-v1',
    'n_iter': 100,
    'ep_len': 1.,
    'reward_to_go': True,
    'dont_normalize_advantages': False,
    'nn_baseline': False,
    'clear_tb_expt': False,
    'n_experiments': 1,
    'seed': 1,
    'render': False,
    'num_parallel': 1,
}
PG_PARAMS = collections.namedtuple(
    'PG_PARAMS',
    [
        'exp_name',
        'discount',
        'n_experiments',
        'env_name',
        'n_iter',
        'ep_len',
        'reward_to_go',
        'dont_normalize_advantages',
        'nn_baseline',
        'clear_tb_expt',
        'learning_rate',
        'n_layers',
        'size',
        'batch_size',
        'seed',
        'render',
        'num_parallel',
    ],
)


def main():
    tb.clear_expts()
    pool = mlp.Pool(processes=NUM_PROCESSES)
    pool.map(_run_setting, _sample_settings(num=NUM_SETTINGS))


def _run_setting(iternum_and_pg_params):
    iternum, pg_params = iternum_and_pg_params
    print '\n\n----------------------------------------------------'
    print '\t\tHyperparam search: `{}`'.format(iternum)
    print '----------------------------------------------------'
    print 'Param setting: {}'.format(pg_params)
    train_pg.train_wrapper(params=pg_params)


def _sample_settings(
    num,
    default_params=DEFAULT_PARAMS,
    hyperparam_choices=HYPERPARAM_CHOICES,
):
    return [
        (i, _sample_one_setting(iternum=i))
        for i in xrange(1, num+1)
    ]

def _sample_one_setting(
        iternum, default_params=DEFAULT_PARAMS,
        hyperparam_choices=HYPERPARAM_CHOICES):
    sampled_hyp_params = {
        param: random.choice(choices)
        for param, choices in hyperparam_choices.iteritems()
    }
    exp_name = 'hs-{env_name}-{i}-{setting}'.format(
        env_name=EXPNAME_SHORT_FORM,
        i=iternum,
        setting='-'.join(
            '{}_{}'.format(HYPERPARAM_SHORT_FORM[param], val)
            for param, val in sampled_hyp_params.iteritems()
        ),
    )
    params = sampled_hyp_params.copy()
    params.update(default_params)
    params['exp_name'] = exp_name
    print params
    return PG_PARAMS(**params)


if __name__ == '__main__':
    main()
