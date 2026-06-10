from __future__ import annotations

import torch
from torch import nn


class CombinedSaliencyLoss(nn.Module):
    def __init__(self, kld_weight: float = 1.0, bce_weight: float = 0.1, mse_weight: float = 0.0) -> None:
        super().__init__()
        self.kld_weight = kld_weight
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        logits: torch.Tensor,
        target_prob: torch.Tensor,
        target_max: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        flat_logits = logits.flatten(start_dim=1)
        log_probs = torch.log_softmax(flat_logits, dim=1).view_as(logits)
        kld = torch.sum(target_prob * (torch.log(target_prob + 1e-8) - log_probs), dim=(1, 2, 3)).mean()
        bce = self.bce(logits, target_max)
        mse = self.mse(torch.sigmoid(logits), target_max)
        loss = self.kld_weight * kld + self.bce_weight * bce + self.mse_weight * mse
        return loss, {"kld": float(kld.detach()), "bce": float(bce.detach()), "mse": float(mse.detach())}
