# -*- coding: utf-8 -*-
"""Code to behavior clone an expert and save the cloned policy.

Sample run::

    python bclone --rollout hopper_rollout.pkl \
        --output hopper_bc.pkl \
        --epochs 1000
"""
from __future__ import division
from __future__ import unicode_literals

import argparse

import torch as tc
from torch.utils import data as tc_data_utils

import pandas as pd
import tqdm

from matplotlib import pyplot as plt

import model_utils

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize']= (15, 15)

pd.options.display.max_colwidth = 75
pd.options.display.width = 150
pd.options.display.max_columns = 40

########################################################################
#                           Models
########################################################################

def fc2layer_model(D_in, D_out, H=100):
    model = tc.nn.Sequential(
        tc.nn.Linear(D_in, H),
        tc.nn.ReLU(),
        tc.nn.Linear(H, H),
        tc.nn.ReLU(),
        tc.nn.Linear(H, D_out),
    )
    loss_fn = tc.nn.MSELoss()
    optimizer = tc.optim.Adam(params=model.parameters(), lr=1e-3)
    print 'Model:', model, '\nLoss fn:', loss_fn, '\nOptimizer:', optimizer
    return {
        'model': model,
        'loss_fn': loss_fn,
        'optimizer': optimizer,
    }

#: Set this to the model you want to build.
MODEL_FUNC = fc2layer_model

########################################################################
#                       Training
########################################################################

def main():
    cmd_opts = _parse_args()
    train(
        rollout_fl=cmd_opts.rollout_file,
        epochs=cmd_opts.epochs,
        output_model_file=cmd_opts.output_model_file,
    )
    print (
        'In order to see the policy, run: \n'
        'python run_expert.py {m}  <env_name> --render '
        '--user_policy.'.format(m=cmd_opts.output_model_file)
    )
    print (
        'Expert names: \nHopper-v1 Ant-v1 '
        'HalfCheetah-v1 Humanoid-v1 Reacher-v1 Walker2d-v1'
    )


def _parse_args():
    """Parse command line arguments"""
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument(
        '-i',
        '--rollout',
        dest='rollout_file',
        help='File containing expert rollouts',
        type=str,
        required=True,
    )
    cmd_parser.add_argument(
        '-o',
        '--output',
        dest='output_model_file',
        type=str,
        help=(
            'Where to save the trained model pickle file.'
        ),
        required=True,
    )
    cmd_parser.add_argument(
        '-e',
        '--epochs',
        dest='epochs',
        type=int,
        help='Number of epochs. Default 500.',
        default=500,
    )
    return cmd_parser.parse_args()


def train(rollout_fl, epochs, output_model_file, model_func=MODEL_FUNC):
    rolls = model_utils.load_rollouts(fl=rollout_fl)
    dataloader = model_utils.RollDataLoader(rolls=rolls)
    D_in, D_out = model_utils.get_policy_dims(rolls=rolls)
    model_elems = model_func(
        D_in=D_in,
        D_out=D_out,
    )
    losses = model_utils.train_model(
        dataloader=dataloader,
        epochs=epochs,
        **model_elems
    )
    model_utils.plot_loss(losses=losses)
    model_utils.save_model(
        path=output_model_file,
        model=model_elems['model'],
    )


def study_input(rolls):
    obs_df = pd.DataFrame(rolls.obs)
    print 'Observation df: \n', obs_df
    print 'Observation always zero features: \n'
    print_num_zero_inds(df=obs_df)

    axns_df = pd.DataFrame(rolls.axns)
    _ = axns_df.hist()
    plt.title('Histogram of action inputs')
    plt.show()


def print_num_zero_inds(df):
    num_non_zeros = (df != 0).sum(axis=0)
    allzeroinds = num_non_zeros[(num_non_zeros==0)].index.values
    print 'allzeroinds: ', allzeroinds, '\ncnt: ', len(allzeroinds)


if __name__ == '__main__':
    main()