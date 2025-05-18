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

