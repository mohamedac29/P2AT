from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

from configs import config


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    Useful for tracking loss, accuracy, etc. during training.
    """

    def __init__(self):
        """Initialize all metrics as None"""
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        """Initialize the meter with initial value and weight"""
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        """Update the meter with new value"""
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        """Add a new value with given weight"""
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        """Return current value"""
        return self.val

    def average(self):
        """Return running average"""
        return self.avg
    

def create_logger(cfg, cfg_name, phase='train'):
    """
    Create logger for tracking training progress and saving logs.
    Args:
        cfg: configuration object
        cfg_name: configuration filename
        phase: 'train' or other phase identifier
    Returns:
        logger: logging object
        final_output_dir: path for saving checkpoints
        tensorboard_log_dir: path for tensorboard logs
    """
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # Create output directory if it doesn't exist
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    # Create dataset-specific output directory
    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with timestamp
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    
    # Set up logging configuration
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Also log to console
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # Create directory for TensorBoard logs
    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)