import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == "none":
            loss = loss
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class MCRMSELoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Compute RMSE per feature
        loss_per_feature = torch.sqrt(self.mse(y_pred, y_true) + self.eps)

        # Compute mean RMSE over all features
        loss = loss_per_feature.mean(dim=0)

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()


class WeightedMSELoss(nn.Module):
    def __init__(self, task_num=2):
        super(WeightedMSELoss, self).__init__()
        self.task_num = task_num
        self.smoothl1loss = nn.SmoothL1Loss(reduction="mean")

    def forward(self, preds, ans, log_vars):
        loss = 0
        for i in range(self.task_num):
            precision = torch.exp(-log_vars[i])
            diff = self.smoothl1loss(preds[:, i], ans[:, i])
            loss += torch.sum(precision * diff + log_vars[i], -1)

        loss = 0.5 * loss

        return loss


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 0.0, target_weights=[0.3, 0.7]):
        """
        https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/smooth_l1_loss.py
        """

        super().__init__()
        self.beta = beta
        self.target_weights = torch.Tensor(target_weights).cuda()

    def forward(self, pred, target):
        if self.beta < 1e-5:
            loss = torch.abs(pred - target)
        else:
            n = torch.abs(pred - target)
            cond = n < self.beta
            loss = torch.where(cond, torch.pow(0.5 * n, 2) / self.beta, n - 0.5 * self.beta)

        loss = torch.sum(loss * self.target_weights)
        return loss


class WeightedRMSELoss(nn.Module):
    def __init__(self, target_weights=[0.3, 0.7], reduction="mean", eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.reduction = reduction
        self.eps = eps
        self.target_weights = torch.Tensor(target_weights).cuda()

    def forward(self, y_pred, y_true):
        # Compute per-element RMSE
        per_element_loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)

        # Apply target weights
        weighted_loss = per_element_loss * self.target_weights

        # Apply reduction
        if self.reduction == "none":
            loss = weighted_loss
        elif self.reduction == "mean":
            loss = weighted_loss.mean()
        elif self.reduction == "sum":
            loss = weighted_loss.sum()

        return loss
