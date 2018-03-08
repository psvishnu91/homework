# -*- coding: utf-8 -*-
import torch as tc


def fc(D_in, D_out, H=100, n_hidden=2):
    layers = (
        [tc.nn.Linear(D_in, H), tc.nn.ReLU()] +
        ([tc.nn.Linear(H, H), tc.nn.ReLU()] * (n_hidden - 1)) +
        [tc.nn.Linear(H, D_out)]
    )
    model = tc.nn.Sequential(*layers)
    loss_fn = tc.nn.MSELoss()
    optimizer = tc.optim.Adam(params=model.parameters(), lr=1e-3)
    print 'Model:', model, '\nLoss fn:', loss_fn, '\nOptimizer:', optimizer
    return {
        'model': model,
        'loss_fn': loss_fn,
        'optimizer': optimizer,
    }