#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1  \
        savefl.pkl \
        --render \
        --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import load_policy

import model_utils
import run_sim


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('output_file',  nargs='?', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--user_policy', action='store_true')
    args = parser.parse_args()

    print('loading and building expert policy')
    if args.user_policy:
        policy_fn = model_utils.load_policy(path=args.expert_policy_file)
    else:
        policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    expert_data = run_sim.run_sim(
        policy_fn=policy_fn,
        envname=args.envname,
        num_rollouts=args.num_rollouts,
        render=args.render,
        max_timesteps=args.max_timesteps,
    )
    expert_data['envname'] = args.envname
    if args.output_file:
        with open(args.output_file, 'w') as outfile:
            pickle.dump(obj=expert_data, file=outfile)



if __name__ == '__main__':
    main()
