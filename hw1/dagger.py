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
        --output-path humanoid_dagger.pkl \
        --capture-dir videos_dagger/
"""
from __future__ import division
from __future__ import unicode_literals

import argparse
import functools
import os

import numpy as np
import tensorflow as tf
import tqdm

import load_policy
import models
import model_utils
import run_sim
import tensorboard as tb
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
    tb.clear_expts()
    tb_expt_global = tb.get_experiment(name='dagger_global')
    sim_kwargs = dict(
        envname=env,
        render=False,
        max_timesteps=MAX_TIMESTEPS,
    )
    print 'Loading expert policy...'
    expert_policy_fn = load_policy.load_policy(filename=expert_path)
    rolls = run_sim.sim_to_rollout(
        policy_fn=expert_policy_fn,
        num_rollouts=init_rollout_sz,
        ** sim_kwargs
    )
    model_elems = model_utils.train(
        rolls=rolls,
        epochs=epochs,
        model_creation_func=MODEL_FUNC,
        tb_expts=[
            tb.get_experiment('dagger_init-{}'.format(env)),
            tb_expt_global,
        ],
        to_plot=False,
    )
    clone_policy_fn = model_utils.model_to_policy(model_elems['model'])
    print 'Saving video of rollout after init training...'
    run_sim.sim_to_rollout(
        policy_fn=clone_policy_fn,
        capture_dir=_get_capture_sub_dir(fld=capture_dir, it=0),
        num_rollouts=1,
        **sim_kwargs
    )
    for i in tqdm.tqdm(range(1, max_dagger_iters+1), desc='dagger_iters'):
        rolls = train_dagger_iter(
            rolls_so_far=rolls,
            model_elems=model_elems,
            epochs=epochs,
            clone_policy_fn=clone_policy_fn,
            expert_policy_fn=expert_policy_fn,
            output_model_path=None,
            tb_expts=[
                tb.get_experiment(
                    name='dagger_iter_{i}-{e}'.format(i=i, e=env),
                ),
                tb_expt_global,
            ],
            sim_kwargs=sim_kwargs,
            iter_num=i,
            rollout_sz=rollout_sz,
            capture_dir=capture_dir,
        )
    print 'Saving the model...'
    model_utils.save_model(
        path=output_model_path,
        model=model_elems['model'],
    )
    return model_elems['model']

def _get_capture_sub_dir(fld, it):
    if not fld:
        return fld
    return os.path.join(fld, str(it))

def train_dagger_iter(
        rolls_so_far, model_elems, epochs, clone_policy_fn,
        expert_policy_fn, sim_kwargs, iter_num, capture_dir,
        tb_expts, rollout_sz, output_model_path=None):
    print 'Run the trained policy to get obs...'
    roll_data = run_sim.run_sim(
        policy_fn=clone_policy_fn,
        num_rollouts=rollout_sz,
        **sim_kwargs
    )
    for each_expt in tb_expts:
        each_expt.add_scalar_dict(
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
                [
                    expert_policy_fn(o[None, :])
                    for o in roll_data['observations']
                ],
                axis=0,
            ),
        ),
    )
    print 'Training on the aggregated dataset...'
    model_utils.train(
        rolls=agg_rolls,
        epochs=epochs,
        model_elems=model_elems,
        tb_expts=tb_expts,
        to_plot=False,
    )
    print 'Saving video after dagger iter `{}`...'.format(iter_num)
    run_sim.sim_to_rollout(
        policy_fn=model_utils.model_to_policy(model_elems['model']),
        capture_dir=_get_capture_sub_dir(fld=capture_dir, it=iter_num),
        num_rollouts=1,
        **sim_kwargs
    )
    return agg_rolls


if __name__ == '__main__':
    main()
