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
import functools

import pandas as pd

from matplotlib import pyplot as plt

import models
import model_utils

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize']= (15, 15)

pd.options.display.max_colwidth = 75
pd.options.display.width = 150
pd.options.display.max_columns = 40


#: Set this to the model you want to build.
MODEL_FUNC = functools.partial(models.fc,  H=100, n_hidden=2)

########################################################################
#                       Training
########################################################################

def main():
    cmd_opts = _parse_args()
    with tf.Session():
        tf_util.initialize()
        train_wrapper(
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


def train_wrapper(rollout_fl, output_model_file, epochs):
    rolls = model_utils.load_rollouts(fl=rollout_fl)
    model = model_utils.train(
        rolls=rolls,
        epochs=epochs,
        model_creation_func=MODEL_FUNC,
    )['model']
    model_utils.save_model(path=output_model_file, model=model)


def study_input(rolls):
    obs_df = pd.DataFrame(rolls.obs)
    print 'Observation df: \n', obs_df
    print 'Observation always zero features: \n'
    print_num_zero_inds(df=obs_df)

    axns_df = pd.DataFrame(rolls.axns)
    axns_df.hist()
    plt.title('Histogram of action inputs')
    plt.show()


def print_num_zero_inds(df):
    num_non_zeros = (df != 0).sum(axis=0)
    allzeroinds = num_non_zeros[(num_non_zeros==0)].index.values
    print 'allzeroinds: ', allzeroinds, '\ncnt: ', len(allzeroinds)


if __name__ == '__main__':
    main()
