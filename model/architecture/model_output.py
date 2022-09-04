from typing import Dict, Optional
import pytorch_lightning as pl
import torch
from torch import nn as nn
from torchmetrics import Metric


class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        basis_set_size: int = 36,
        metrics: Optional[Dict[str, Metric]] = None,
        target_property: Optional[str] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.basis_set_size = basis_set_size
        self.metrics = nn.ModuleDict(metrics)
        self.constraints = []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property], self.basis_set_size
        )
        return loss

    def calculate_metrics(self, pred, target):
        metrics = {
            metric_name: metric(pred[self.name], target[self.target_property])
            for metric_name, metric in self.metrics.items()
        }
        return metrics