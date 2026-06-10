from __future__ import annotations

import torch


def probability_map(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.flatten(start_dim=1), dim=1).view_as(logits)


def sigmoid_map(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def kld_metric(logits: torch.Tensor, target_prob: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits.flatten(start_dim=1), dim=1).view_as(logits)
    kld = torch.sum(target_prob * (torch.log(target_prob + 1e-8) - log_probs), dim=(1, 2, 3)).mean()
    return float(kld.detach())


def mse_metric(logits: torch.Tensor, target: torch.Tensor, use_probability: bool) -> float:
    pred = probability_map(logits) if use_probability else sigmoid_map(logits)
    mse = torch.mean((pred - target) ** 2)
    return float(mse.detach())


def cc_metric(logits: torch.Tensor, target: torch.Tensor, use_probability: bool) -> float:
    pred = probability_map(logits) if use_probability else sigmoid_map(logits)
    pred = pred.flatten(start_dim=1)
    target = target.flatten(start_dim=1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (pred * target).sum(dim=1)
    denominator = torch.sqrt((pred.pow(2).sum(dim=1) + 1e-8) * (target.pow(2).sum(dim=1) + 1e-8))
    cc = numerator / denominator
    return float(cc.mean().detach())
