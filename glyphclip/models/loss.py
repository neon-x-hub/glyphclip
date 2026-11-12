"""Temperature-scaled InfoNCE (NT-Xent) contrastive loss function."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric Temperature-scaled InfoNCE (NT-Xent) Loss.

    Optimizes alignment between visual query embeddings z_v and canonical
    glyph embeddings z_c, treating all other canonical candidates in the
    batch as negative samples.
    """

    def __init__(self, init_temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        self.learnable = learnable
        initial_log_tau = math.log(init_temperature)
        if learnable:
            self.log_tau = nn.Parameter(torch.tensor(initial_log_tau, dtype=torch.float32))
        else:
            self.register_buffer("log_tau", torch.tensor(initial_log_tau, dtype=torch.float32))

    @property
    def temperature(self) -> float:
        """Current temperature tau."""
        return float(torch.exp(self.log_tau).clamp(min=1e-4, max=1.0).item())

    def forward(
        self,
        z_v: torch.Tensor,
        z_c: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute symmetric InfoNCE loss between batch pairs.

        Args:
            z_v: Visual query embeddings [B, d], normalized.
            z_c: Canonical glyph embeddings [B, d], normalized.

        Returns:
            Tuple of (loss_scalar, metrics_dict).
        """
        batch_size = z_v.size(0)
        if batch_size == 0:
            raise ValueError("Empty batch provided to InfoNCELoss.")

        # Clamp temperature to prevent numerical instability
        tau = torch.exp(self.log_tau).clamp(min=1e-4, max=1.0)

        # Pairwise cosine similarity matrix: [B, B]
        # Since z_v and z_c are L2-normalized, dot product equals cosine similarity
        sim_matrix = torch.matmul(z_v, z_c.T) / tau

        labels = torch.arange(batch_size, device=z_v.device, dtype=torch.long)

        # Symmetric cross entropy
        loss_v2c = F.cross_entropy(sim_matrix, labels)
        loss_c2v = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss_v2c + loss_c2v) / 2.0

        # Compute diagnostic metrics
        with torch.no_grad():
            pos_sim = torch.diag(torch.matmul(z_v, z_c.T)).mean().item()
            # Off-diagonal negative similarities
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_v.device)
            neg_sim = (torch.matmul(z_v, z_c.T)[mask].mean().item()) if batch_size > 1 else 0.0
            pred = sim_matrix.argmax(dim=-1)
            accuracy = (pred == labels).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "temperature": tau.item(),
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "in_batch_top1_acc": accuracy,
        }

        return loss, metrics
