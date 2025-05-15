# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config

class FullModel(nn.Module):
    """
    A wrapper class that combines the main model with semantic and boundary loss functions.
    This class handles the forward pass and computes all losses and metrics.
    """
    
    def __init__(self, model, sem_loss, b_loss):
        """
        Initialize the FullModel with:
        - model: the main neural network model
        - sem_loss: loss function for semantic segmentation
        - b_loss: loss function for boundary prediction
        """
        super(FullModel, self).__init__()
        self.model = model
        self.sem_loss = sem_loss
        self.b_loss = b_loss

    def pixel_acc(self, pred, label):
        """
        Calculate pixel-wise accuracy for semantic segmentation.
        Args:
            pred: model predictions (logits)
            label: ground truth labels
        Returns:
            Accuracy percentage
        """
        _, preds = torch.max(pred, dim=1)  # Get predicted class indices
        valid = (label >= 0).long()  # Identify valid pixels (ignore label < 0)
        acc_sum = torch.sum(valid * (preds == label).long())  # Count correct predictions
        pixel_sum = torch.sum(valid)  # Count total valid pixels
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)  # Calculate accuracy
        return acc

    def forward(self, inputs, labels, b_gt, *args, **kwargs):
        """
        Forward pass with loss computation.
        Args:
            inputs: input images
            labels: semantic segmentation ground truth
            b_gt: boundary prediction ground truth
        Returns:
            Tuple containing:
            - total loss
            - model outputs
            - accuracy
            - individual loss components
        """
        # Get model outputs (multiple outputs expected)
        outputs = self.model(inputs, *args, **kwargs)
        
        # Calculate accuracy from second-to-last output
        acc = self.pixel_acc(outputs[-2], labels)
        
        # Calculate semantic loss for all outputs except last one
        s_loss = self.sem_loss(outputs[:-1], labels)
        
        # Calculate boundary loss from last output
        b_loss = self.b_loss(outputs[-1], b_gt)

        # Create a special label for semantic-boundary loss:
        # Use original labels where boundary confidence > 0.8, otherwise ignore
        filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
        b_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:]) > 0.8, labels, filler)
        
        # Calculate semantic-boundary loss
        sb_loss = self.sem_loss(outputs[-2], b_label)
        
        # Combine all losses
        loss = s_loss + b_loss + sb_loss

        return torch.unsqueeze(loss, 0), outputs[:-1], acc, [s_loss, b_loss]


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

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calculate confusion matrix for semantic segmentation evaluation.
    Args:
        label: ground truth labels
        pred: model predictions
        size: original image size (before padding)
        num_class: number of classes
        ignore: ignore label value
    Returns:
        confusion_matrix: num_class x num_class numpy array
    """
    # Convert predictions to numpy and get class indices
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    
    # Get ground truth (cropped to original size)
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    # Ignore specified pixels
    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    # Calculate confusion matrix using bincount
    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    # Populate confusion matrix
    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix

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