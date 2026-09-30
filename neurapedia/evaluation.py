"""Research metric primitives. Inputs need independent reference labels."""
from __future__ import annotations
import numpy as np


def segmentation_metrics(prediction, reference) -> dict:
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or not prediction.size:
        raise ValueError("Masks must have identical non-empty shapes")
    if not np.isin(prediction, [0, 1]).all() or not np.isin(reference, [0, 1]).all():
        raise ValueError("Binary masks are required")
    p, r = prediction.astype(bool), reference.astype(bool)
    intersection, union = int((p & r).sum()), int((p | r).sum())
    size = int(p.sum() + r.sum())
    return {"dice": 2 * intersection / size if size else 1.0,
            "iou": intersection / union if union else 1.0,
            "false_positive_voxels": int((p & ~r).sum()),
            "false_negative_voxels": int((~p & r).sum()),
            "empty_reference": not bool(r.any())}


def probability_metrics(probabilities, reference, bins: int = 10) -> dict:
    p, r = np.asarray(probabilities, dtype=float), np.asarray(reference)
    if p.shape != r.shape or not p.size or not 1 <= bins <= 100:
        raise ValueError("Invalid probability evaluation shape or bins")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isin(r, [0, 1]).all():
        raise ValueError("Finite probabilities [0,1] and binary labels are required")
    ece = 0.0
    for i in range(bins):
        selected = (p >= i / bins) & ((p <= 1) if i == bins - 1 else (p < (i + 1) / bins))
        if selected.any():
            ece += selected.mean() * abs(float(p[selected].mean() - r[selected].mean()))
    return {"brier_score": float(np.mean((p - r) ** 2)), "ece": float(ece),
            "mean_binary_entropy": float(np.mean(-(p * np.log(np.clip(p, 1e-12, 1))
                                                 + (1-p) * np.log(np.clip(1-p, 1e-12, 1)))))}
