"""

Some simple logging functionality, inspired by rllab's logging.
Assumes that each diagnostic gets logged each iteration

Call logz.configure_output_dir() to start logging to a
tab-separated-values file (some_folder_name/log.txt)

To load the learning curves, you can do, for example

A = np.genfromtxt('/tmp/expt_1468984536/log.txt',delimiter='\t',dtype=None, names=True)
A['EpRewMean']

"""

import copy
import json
import os.path as osp
import shutil
import time
import atexit
import os
import pickle
import subprocess

import numpy as np
import pandas as pd
import tensorflow as tf


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class G:
    output_dir = None
    output_file = None
    first_row = True
    log_headers = []
    log_current_row = {}

def configure_output_dir(d=None):
    """
    Set output directory to d, or to /tmp/somerandomnumber if d is None
    """
    G.output_dir = d or "/tmp/experiments/%i"%int(time.time())
    assert not osp.exists(G.output_dir), "Log dir %s already exists! Delete it first or use a different dir"%G.output_dir
    os.makedirs(G.output_dir)
    G.output_file = open(osp.join(G.output_dir, "log.txt"), 'w')
    atexit.register(G.output_file.close)
    print(colorize("Logging data to %s"%G.output_file.name, 'green', bold=True))

def log_tabular(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    if G.first_row:
        G.log_headers.append(key)
    else:
        assert key in G.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
    assert key not in G.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
    G.log_current_row[key] = val

def save_params(params):
    with open(osp.join(G.output_dir, "params.json"), 'w') as out:
        out.write(json.dumps(params, separators=(',\n','\t:\t'), sort_keys=True))

def pickle_tf_vars():
    """
    Saves tensorflow variables
    Requires them to be initialized first, also a default session must exist
    """
    _dict = {v.name : v.eval() for v in tf.global_variables()}
    with open(osp.join(G.output_dir, "vars.pkl"), 'wb') as f:
        pickle.dump(_dict, f)


def dump_tabular():
    """
    Write all of the diagnostics from the current iteration
    """
    vals = []
    key_lens = [len(key) for key in G.log_headers]
    max_key_len = max(15,max(key_lens))
    keystr = '%'+'%d'%max_key_len
    fmt = "| " + keystr + "s | %15s |"
    n_slashes = 22 + max_key_len
    print("-"*n_slashes)
    for key in G.log_headers:
        val = G.log_current_row.get(key, "")
        if hasattr(val, "__float__"): valstr = "%8.3g"%val
        else: valstr = val
        print(fmt%(key, valstr))
        vals.append(val)
    print("-"*n_slashes)
    if G.output_file is not None:
        if G.first_row:
            G.output_file.write("\t".join(G.log_headers))
            G.output_file.write("\n")
        G.output_file.write("\t".join(map(str,vals)))
        G.output_file.write("\n")
        G.output_file.flush()
    G.log_current_row.clear()
    G.first_row=False

#============================================================================================#
# Tensorboard Plotting
#============================================================================================#

def log_scalar(name, val, tb_expt, history_dict):
    # Don't mutate the input. Mutation is evil.
    history_dict = copy.deepcopy(history_dict)
    log_tabular(name, val)
    history_dict.setdefault(name, []).append(val)
    tb_expt.add_scalar_dict({name: float(val)})
    return history_dict


def log_actions(actions, tb_expt):
    for j in xrange(actions.shape[1]):
        tb_expt.add_histogram_value(
            name='axn/{j}'.format(j=j),
            hist=actions[:, j].tolist(),
            tobuild=True,
        )


def log_value_scalars(
    itr, start_time, returns, loss_val, ep_lengths, timesteps_this_batch,
    total_timesteps, tb_expt, history_dict,
):
    hd = history_dict
    hd = log_scalar('Time', time.time() - start_time, tb_expt, hd)
    hd = log_scalar('Iteration', itr, tb_expt, hd)
    hd = log_scalar('loss', loss_val, tb_expt, hd)
    hd = log_scalar('Return/Avg', np.mean(returns), tb_expt, hd)
    hd = log_scalar('Return/Std', np.std(returns), tb_expt, hd)
    hd = log_scalar('Return/Max', np.max(returns), tb_expt, hd)
    hd = log_scalar('Return/Min', np.min(returns), tb_expt, hd)
    hd = log_scalar('EpLen/Mean', np.mean(ep_lengths), tb_expt, hd)
    hd = log_scalar('EpLen/Std', np.std(ep_lengths), tb_expt, hd)
    hd = log_scalar('Timesteps/ThisBatch', timesteps_this_batch, tb_expt, hd)
    hd = log_scalar('Timesteps/SoFar', total_timesteps, tb_expt, hd)
    return hd


def plot_tb_avg_history(tb_avg_expt, history_dicts):
    avg_df = pd.DataFrame(history_dicts[0])
    for hd in history_dicts[1:]:
        avg_df += pd.DataFrame(hd)
    avg_df /= len(history_dicts)
    for field in avg_df.columns:
        for val in avg_df[field]:
            tb_avg_expt.add_scalar_dict({field: float(val)})
