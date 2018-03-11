# -*- coding: utf-8 -*-
"""Code to learn a policy from an expert with DAGGER algorithm.

Sample run::

    python dagger.py \
        --expert experts/Humanoid-v1.pkl \
        --env Humanoid-v1 \
        --init-rollout-sz 20000 \
        --init-epochs 500 \
        --rollout-sz 7500 \
        --epochs 200 \
        --max-dagger-iters 3 \
        --output-path humanoid_dagger.pkl
"""
from __future__ import division
from __future__ import unicode_literals

import argparse
import copy
import functools

import numpy as np
import tensorflow as tf
import tqdm

import load_policy
import models
import model_utils
import run_sim
import tensorboard
import tf_util


#: Set this to the model you want to build.
MODEL_FUNC = functools.partial(models.fc,  H=100, n_hidden=2)
MAX_TIMESTEPS = 1000
########################################################################
#                       Training
########################################################################


def main():
    cmd_opts = _parse_args()
    print (
        'Expert names: \nHopper-v1 Ant-v1 '
        'HalfCheetah-v1 Humanoid-v1 Reacher-v1 Walker2d-v1'
    )
    with tf.Session():
        tf_util.initialize()
        model = train_wrapper(
            expert_path=cmd_opts.expert,
            env=cmd_opts.env,
            init_rollout_sz=cmd_opts.init_ro_sz,
            rollout_sz=cmd_opts.ro_sz,
            init_epochs=cmd_opts.init_epochs,
            epochs=cmd_opts.epochs,
            max_dagger_iters=cmd_opts.max_dagger_iters,
            output_model_path=cmd_opts.output_path,
            capture_dir=cmd_opts.capture_dir,
        )
        if cmd_opts.to_render:
            run_sim.run_sim(
                policy_fn=model_utils.model_to_policy(model=model),
                envname=cmd_opts.env,
                num_rollouts=3,
                render=True,
                max_timesteps=None,
            )


def _parse_args():
    """Parse command line arguments"""
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument(
        '--expert',
        dest='expert',
        help='Path to the expert to mimic.',
        type=str,
        required=True,
    )
    cmd_parser.add_argument(
        '--env',
        dest='env',
        help='Environment name to mimic.',
        type=str,
        required=True,
    )
    cmd_parser.add_argument(
        '-ir',
        '--init-rollout-sz',
        dest='init_ro_sz',
        help=(
            'The batch size for the initial rollout used for '
            'policy initializations.'
        ),
        type=int,
        required=True,
    )
    cmd_parser.add_argument(
        '-r',
        '--rollout-sz',
        dest='ro_sz',
        type=int,
        help=(
            'The size of rollouts to use for aggregation training.'
        ),
        required=True,
    )
    cmd_parser.add_argument(
        '-ie',
        '--init-epochs',
        dest='init_epochs',
        type=int,
        help='Number of epochs for initial training. Default 500.',
        default=500,
    )
    cmd_parser.add_argument(
        '-e',
        '--epochs',
        dest='epochs',
        type=int,
        help='Number of epochs. Default 200.',
        default=200,
    )
    cmd_parser.add_argument(
        '--max-dagger-iters',
        dest='max_dagger_iters',
        type=int,
        help='Max number dagger loops.',
        default=10,
    )
    cmd_parser.add_argument(
        '--output-path',
        dest='output_path',
        type=str,
        help='Path to save the cloned model.',
        required=True,
    )
    cmd_parser.add_argument(
        '--no-render',
        dest='to_render',
        action='store_false',
        help='Whether to render the model at the end to the screen.',
        default=True,
    )
    cmd_parser.add_argument(
        '--capture-dir',
        dest='capture_dir',
        type=str,
        help='Where to save the video after every dagger iter.',
        default=None,
    )
    return cmd_parser.parse_args()


def train_wrapper(
    expert_path,
    env,
    init_rollout_sz,
    rollout_sz,
    init_epochs,
    epochs,
    max_dagger_iters,
    output_model_path,
    capture_dir,
):
    tensorboard.clear_expts()
    tb_expt = tensorboard.get_experiment('dagger_init-{}'.format(env))
    sim_kwargs = dict(
        envname=env,
        num_rollouts=init_rollout_sz,
        render=False,
        max_timesteps=MAX_TIMESTEPS,
        capture_dir=capture_dir,
    )
    print 'Loading expert policy...'
    expert_policy_fn = load_policy.load_policy(filename=expert_path)
    rolls = run_sim.sim_to_rollout(
        policy_fn=expert_policy_fn,
        ** sim_kwargs
    )
    model_elems = model_utils.train(
        rolls=rolls,
        epochs=epochs,
        model_creation_func=MODEL_FUNC,
        tb_expt=tb_expt,
        to_plot=False,
    )
    clone_policy_fn = model_utils.model_to_policy(model_elems['model'])
    for i in tqdm.tqdm(range(max_dagger_iters), desc='dagger_iters'):
        sim_kwargs = copy.deepcopy(x=sim_kwargs)
        rolls = train_dagger_iter(
            rolls_so_far=rolls,
            model_elems=model_elems,
            epochs=epochs,
            clone_policy_fn=clone_policy_fn,
            expert_policy_fn=expert_policy_fn,
            output_model_path=None,
            tb_expt=tensorboard.get_experiment(
                name='dagger_iter_{i}-{e}'.format(i=i, e=env),
            ),
            sim_kwargs=sim_kwargs,
        )
    print 'Saving the model...'
    model_utils.save_model(
        path=output_model_path,
        model=model_elems['model'],
    )
    return model_elems['model']


def train_dagger_iter(
        rolls_so_far, model_elems, epochs, clone_policy_fn,
        expert_policy_fn, sim_kwargs, output_model_path=None,
        tb_expt=None):
    print 'Run the trained policy to get obs...'
    roll_data = run_sim.run_sim(
        policy_fn=clone_policy_fn,
        **sim_kwargs
    )
    expert_policy_fn(roll_data['observations'][0][None, :])
    tb_expt.add_scalar_dict(
        {
            'return/mean': roll_data['returns']['mean'],
            'return/stddev': roll_data['returns']['std'],
        },
    )
    print 'Predicting the expert for the rollout...'
    agg_rolls = model_utils.concat_rollouts(
        rolls_so_far,
        model_utils.Rollout(
            obs=roll_data['observations'],
            axns=np.concatenate(
                [expert_policy_fn(o[None, :])
                 for o in roll_data['observations']],
                axis=0,
            ),
        ),
    )
    print 'Training on the aggregated dataset...'
    model_utils.train(
        rolls=agg_rolls,
        epochs=epochs,
        model_elems=model_elems,
        tb_expt=tb_expt,
        to_plot=False,
    )
    return agg_rolls


if __name__ == '__main__':
    main()
