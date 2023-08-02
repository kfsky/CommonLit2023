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
