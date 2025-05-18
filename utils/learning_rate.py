from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    """
    Adjust learning rate according to polynomial decay schedule.
    Args:
        optimizer: the optimizer whose learning rate should be changed
        base_lr: initial learning rate
        max_iters: maximum number of iterations
        cur_iters: current iteration number
        power: exponent for polynomial decay
        nbb_mult: multiplier for bias parameters' learning rate
    Returns:
        The new learning rate
    """
    # Calculate new learning rate
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    
    # Update learning rate for main parameters
    optimizer.param_groups[0]['lr'] = lr
    
    # Update learning rate for bias parameters if they're in a separate group
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr