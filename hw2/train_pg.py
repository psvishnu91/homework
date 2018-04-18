"""Main script which implements policy gradients.

.. note::

    If you want to visualize with tensorboard run

    .. code-block:: bash

        pip install pycrayon
        docker pull alband/crayon
        # Tensorboard will be served on localhost:9118
        docker run -p 9118:8888 -p 9119:8889 --name crayon alband/crayon

    Each expt run will show up on tb as
    ``{exp_name}-{i};{datetime}_{hostname}`` where i is the expt number
    or ``avg`` for the average of all the runs.

    If you don't want to use tensorboard, just run the script with the
    ``--no_tb`` cmd line arg.
"""
import argparse
import copy
import multiprocessing
import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import path_return
import tensorboard_pycrayon as tb
import time
import inspect


POLICY_VAR_SCOPE = 'policy'
VALUE_FN_VAR_SCOPE = 'value_fn'
TODO = 'todo'


#============================================================================================#
# Utilities
#============================================================================================#

def build_mlp(
    input_placeholder,
    output_size,
    scope,
    n_layers=2,
    size=64,
    activation=tf.nn.relu,
    output_activation=None,
):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units.
    #
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #========================================================================================#

    with tf.variable_scope(scope):
        hidden = tf.layers.dense(
            inputs=input_placeholder,
            units=size,
            activation=activation,
            trainable=True,
            name='dense_1',
        )
        for layer_num in range(1, n_layers):
            hidden = tf.layers.dense(
                inputs=hidden,
                units=size,
                activation=activation,
                trainable=True,
                name='dense_{}'.format(layer_num + 1),
            )
        output = tf.layers.dense(
            inputs=hidden,
            units=output_size,
            activation=output_activation,
            trainable=True,
            name='dense_output',
        )
    return output


def pathlength(path):
    return len(path['reward'])


#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(
    exp_name='',
    env_name='CartPole-v0',
    n_iter=100,
    gamma=1.0,
    min_timesteps_per_batch=1000,
    max_path_length=None,
    learning_rate=5e-3,
    reward_to_go=True,
    animate=True,
    logdir=None,
    normalize_advantages=True,
    nn_baseline=False,
    seed=0,
    # network arguments
    n_layers=1,
    size=32,
    activation=tf.nn.relu,
    use_tensorboard=True,
):
    # This will be returned by this function.  This will be
    # used to compute the average losses across expts to be plotted
    # in tensorboard.
    # Example:
    #   {
    #       # List lengths num_iteraions
    #       "loss": [1,1,2...,],
    #       "Return/Average": [3,9,2...,],
    #       "Time": [1,2,3...,],
    #   }
    history_dict = {}
    start = time.time()
    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {
        k: locals_[k] if k in locals_ else None
        for k in args
        if k != 'activation'
    }
    logz.save_params(params)
    tb_expt = tb.get_experiment(name=exp_name) if use_tensorboard else None
    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    if not max_path_length:
        max_path_length = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    #========================================================================================#
    # Notes on notation:
    #
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    #
    # Prefixes and suffixes:
    # ob - observation
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    #
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    #
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    #========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name='ob', dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None, 1], name='ac', dtype=tf.int32)
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name='ac', dtype=tf.float32)

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)


    #========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    #
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken,
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the
    #      policy network output ops.
    #
    #========================================================================================#

    # num samples in this batch N*T
    n = tf.shape(sy_ob_no)[0]
    policy_out_op = build_mlp(
        input_placeholder=sy_ob_no,
        output_size=ac_dim,
        scope=POLICY_VAR_SCOPE,
        n_layers=n_layers,
        size=size,
        activation=activation,
        output_activation=None,
    )
    if discrete:
        # FORWARDPROP
        sy_logits_na = policy_out_op
        sy_sampled_ac = tf.multinomial(
            logits=sy_logits_na,
            num_samples=1,
            name='ac_sampling',
        )
        # BACKPROP
        # This should use the actions passed in. This is because
        # we first sample the actions while stepping through the
        # env. Then later we compute log_prob of the actions as a batch
        # from the placeholder.
        sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.squeeze(sy_ac_na),
            logits=sy_logits_na,
        )
    else:
        # FORWARDPROP
        sy_mean = policy_out_op
        # logstd should just be a trainable variable, not a network output.
        with tf.variable_scope(POLICY_VAR_SCOPE):
            sy_logstd = tf.Variable(tf.random_normal([1, ac_dim]))
        sy_sampled_ac = (
            sy_mean +
            tf.exp(sy_logstd) * tf.random_normal(shape=[n, ac_dim])
        )
        # BACKPROP
        # This should use the actions passeds in.
        # Hint: Use the log probability under a multivariate gaussian.
        # log(( 1/ sqrt(2 pi E) ) exp(-0.5 (x-mu).T E-1 (x-mu)) )
        variance_inv = 1 / tf.pow(tf.exp(sy_logstd), 2)
        sy_logprob_n = tf.reduce_sum(
            (0.5) * tf.pow((sy_ac_na - sy_mean), 2) * variance_inv,
            axis=1,
        )

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#

    # Loss function that we'll differentiate to get the policy gradient.
    with tf.variable_scope('loss'):
        loss_op = tf.reduce_mean(tf.multiply(sy_logprob_n, sy_adv_n))
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_prediction = tf.squeeze(
            build_mlp(
                input_placeholder=sy_ob_no,
                output_size=1,
                scope=VALUE_FN_VAR_SCOPE,
                n_layers=n_layers,
                size=size,
                activation=tf.nn.relu,
                output_activation=None,
            ),
        )
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        # YOUR_CODE_HERE
        baseline_update_op = TODO


    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101



    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                ac = ac[0]
                acs.append(ac)
                ob, rew, done, _ = env.step(ac[0] if discrete else ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {
                'observation' : np.array(obs),
                'reward' : np.array(rewards),
                'action' : np.array(acs),
            }
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above).
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t.
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
        #       entire trajectory (regardless of which time step the Q-value should be for).
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above.
        #
        #====================================================================================#

        # YOUR_CODE_HERE
        q_n = path_return.compute_Qs(
            paths=paths,
            gamma=gamma,
            reward_to_go=reward_to_go,
        )

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            b_n = TODO
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            # YOUR_CODE_HERE
            adv_std = adv_n.std()
            adv_std = adv_std if adv_std > 0. else 1.
            adv_n = (adv_n - adv_n.mean()) / adv_std


        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next iteration. Use the
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE
            pass

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.

        # YOUR_CODE_HERE
        loss_val, _ = sess.run(
            fetches=[loss_op, update_op,],
            feed_dict={sy_ob_no: ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n},
        )
        # Log diagnostics
        logz.log_actions(actions=ac_na, tb_expt=tb_expt)
        history_dict = logz.log_value_scalars(
            itr=itr,
            start_time=start,
            returns=[path['reward'].sum() for path in paths],
            loss_val=loss_val,
            ep_lengths=[pathlength(path) for path in paths],
            timesteps_this_batch=timesteps_this_batch,
            total_timesteps=total_timesteps,
            tb_expt=tb_expt,
            history_dict=history_dict,
        )
        logz.dump_tabular()
        logz.pickle_tf_vars()
    return history_dict


def main():
    params = parse_args()
    if not(os.path.exists('data')):
        os.makedirs('data')
    if params.use_tensorboard and params.clear_tb_expt:
        tb.clear_expts()
    train_kwargs_list = create_train_kwargs_list(params=params)
    # Don't use multiprocessing for a single process or experiment as
    # multiprocessing often produces arcane error tracebacks.
    if params.num_parallel > 1:
        n_procs = min(params.num_parallel, len(train_kwargs_list))
        map_func = multiprocessing.Pool(processes=n_procs).map
    else:
        map_func = map
    history_dicts = list(map_func(train_PG_star, train_kwargs_list))
    if params.use_tensorboard:
        # Average all the runs performance and plot on tensorboard.
        logz.plot_tb_avg_history(
            tb_avg_expt=tb.get_experiment(
                '{}-{}'.format(params.exp_name, 'avg'),
            ),
            history_dicts=history_dicts,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--n_experiments',
        '-e',
        help=(
            'Number of times to run with different randomization seeds'
            'to average performance'
        ),
        type=int,
        default=1,
    )
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--num_parallel', '-np', type=int, default=1)
    parser.add_argument('--clear_tb_expt', '-ctb', default=False, action='store_true')
    parser.add_argument(
        '--no_tb',
        dest='use_tensorboard',
        help='Whether to not use tensorboard.',
        default=True,
        action='store_false',
    )
    return parser.parse_args()


def create_train_kwargs_list(params):
    """Creates parameters for each expt run with a different random seed.

    :param params: As parsed by :func:`_parse_args`.
    :type params: namedtuple or class

    :returns: Returns a list of different kwargs to be passed into
        :func:`train_PG` for each expt run.
    :rtype: list(dict(str, object))

    Sample Output::

        [
            {
                'exp_name': 'abc-0',
                'env_name': 'Cart-Pole-v0',
                ...
                'seed': 0,
                'use_tensorboard': False,
            },
            {
                'exp_name': 'abc-1',
                'env_name': 'Cart-Pole-v0',
                ...
                'seed': 10,
                'use_tensorboard': False,
            },
        ]
    """
    logdir = '{exp_name}_{env_name}_{time}'.format(
        exp_name=params.exp_name,
        env_name=params.env_name,
        time=time.strftime("%d-%m-%Y_%H-%M-%S"),
    )
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    max_path_length = params.ep_len if params.ep_len > 0 else None
    kwargs_list = []
    for e in range(params.n_experiments):
        seed = params.seed + 10 * e
        print('Running experiment with seed %d' % seed)
        kwargs_list.append(
            dict(
                exp_name='{}-{}'.format(params.exp_name, e),
                env_name=params.env_name,
                n_iter=params.n_iter,
                gamma=params.discount,
                min_timesteps_per_batch=params.batch_size,
                max_path_length=max_path_length,
                learning_rate=params.learning_rate,
                reward_to_go=params.reward_to_go,
                animate=params.render,
                logdir=os.path.join(logdir, '%d' % seed),
                normalize_advantages=not(params.dont_normalize_advantages),
                nn_baseline=params.nn_baseline,
                seed=seed,
                n_layers=params.n_layers,
                size=params.size,
                use_tensorboard=params.use_tensorboard,
            ),
        )
    return kwargs_list


def train_PG_star(train_kwargs):
    return train_PG(**train_kwargs)


if __name__ == "__main__":
    main()
