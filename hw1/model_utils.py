# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import unicode_literals

import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.autograd as ag
import tqdm

from collections import namedtuple
from collections import defaultdict
from itertools import cycle
from itertools import izip

from torch.utils import data as tc_data_utils

IS_CUDA = tc.cuda.is_available()


class Rollout(namedtuple('Rollout', ['obs', 'axns'])):

    def __len__(self):
        return len(self.obs)

class CudableVariable(ag.Variable):

    def __init__(self, data, *args, **kwargs):
        if IS_CUDA:
            data = data.cuda()
        super(self.__class__, self).__init__(data, *args, **kwargs)


class RollDataLoader(object):

    def __init__(
            self, rolls, val_size=0.3, max_len=None, batch_size=5000,
            num_workers=4, **data_loader_kwargs):
        kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
        }
        kwargs.update(data_loader_kwargs)

        rolls = shuffle_rollout(rollout=rolls)
        train_ix = int(len(rolls) * val_size)

        train_rolls = Rollout(
            obs=rolls.obs[:train_ix],
            axns=rolls.axns[:train_ix],
        )
        self.train_dl = tc_data_utils.DataLoader(
            RollDataset(rolls=train_rolls, max_len=max_len),
            **kwargs
        )

        dev_rolls = Rollout(
            obs=rolls.obs[train_ix:],
            axns=rolls.axns[train_ix:],
        )
        self.dev_dl = tc_data_utils.DataLoader(
            RollDataset(rolls=dev_rolls, max_len=max_len),
            **kwargs
        )

    def iter(self):
        for train_data, dev_data in izip(
                self.train_dl,
                cycle(self.dev_dl),
        ):
            yield {
                split: {
                    nm: CudableVariable(val)
                    for nm, val in data.iteritems()
                }
                for split, data in [
                    ('train', train_data),
                    ('dev', dev_data),
                ]
            }

class RollDataset(tc_data_utils.Dataset):

    def __init__(self, rolls, max_len=None):
        self.len = max_len if max_len else len(rolls.obs)
        self.obs = tc.from_numpy(rolls.obs[:self.len]).float()
        self.axns = tc.from_numpy(rolls.axns[:self.len]).float()

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return {'obs': self.obs[i], 'axns': self.axns[i]}


def load_rollouts(fl):
    with open(fl) as infile:
        rollouts = pickle.load(infile)
    obs, axns = rollouts['observations'], np.squeeze(rollouts['actions'])
    print 'obs.shape: ', obs.shape, ', axns.shape, ', axns.shape
    return Rollout(obs=obs, axns=axns)


def plot_loss(losses):
    for split in ('train', 'dev'):
        plt.figure()
        plt.suptitle(split)
        plt.subplot(121)
        plot_split_loss(losses=losses[split])
        plt.subplot(122)
        plot_split_loss(
            losses=losses[split][20:],
            title_suffix='\nAfter first 20 epochs',
        )
        plt.show()


def plot_split_loss(losses, title_suffix=''):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses over epochs.' + title_suffix)


def compute_loss(data, model, loss_fn):
    x = data['obs']
    y = data['axns']
    return loss_fn(model(x), y)


def train_model(
        model, loss_fn, optimizer, dataloader, epochs,
        prev_losses=None):
    if prev_losses:
        losses = copy.deepcopy(x=prev_losses)
    else:
        losses = defaultdict(list)
    epoch_iter =  tqdm.tqdm(range(epochs), desc='epochs')
    for i in epoch_iter:
        for data in dataloader.iter():
            model.zero_grad()
            train_loss = compute_loss(
                data=data['train'],
                model=model,
                loss_fn=loss_fn,
            )
            train_loss.backward()
            optimizer.step()
            losses['train'].append(train_loss.data[0])
            losses['dev'].append(
                compute_loss(
                    data=data['dev'],
                    model=model,
                    loss_fn=loss_fn,
                ).data[0],
            )
            epoch_iter.set_description(
                'Train loss: {tl:,.5f}, Dev loss: {dl:,.5f}'.format(
                    tl=losses['train'][-1],
                    dl=losses['dev'][-1],
                ),
            )
    return losses


def load_model(path):
    with open(path) as infile:
        model = pickle.load(file=infile)
    if IS_CUDA:
        model = model.cuda()
    return model


def load_policy(path):
    model = load_model(path=path)
    def policy(*args):
        args = [
            CudableVariable(data=tc.from_numpy(arg).float())
            for arg in args
        ]
        return model(*args).data.numpy()
    return policy


def save_model(model, path):
    with open(path, 'w') as outfile:
        pickle.dump(obj=model, file=outfile)


def get_policy_dims(rolls):
    """Returns the input, output dims of the policy"""
    return rolls.obs.shape[1], rolls.axns.shape[1]


def shuffle_rollout(rollout):
    ixs = range(len(rollout.obs))
    np.random.shuffle(ixs)
    return Rollout(obs=rollout.obs[ixs], axns=rollout.axns[ixs])
