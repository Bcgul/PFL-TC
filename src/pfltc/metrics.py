"""Regression metrics, always computed on the original scale (degC)."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true_c: np.ndarray, y_pred_c: np.ndarray, names: list[str]) -> dict[str, float]:
    """MSE, RMSE, MAE (degC), MAPE (%), R^2 and bias per target and overall.

    MAPE is computed on temperatures in degC, never on normalised values:
    on normalised targets the denominators are close to zero and MAPE explodes.
    """
    y_true_c = np.asarray(y_true_c, dtype=np.float64)
    y_pred_c = np.asarray(y_pred_c, dtype=np.float64)
    if y_true_c.shape != y_pred_c.shape:
        raise ValueError(f"shape mismatch {y_true_c.shape} vs {y_pred_c.shape}")
    out: dict[str, float] = {}

    def _block(t: np.ndarray, p: np.ndarray, prefix: str) -> None:
        err = p - t
        mse = float(np.mean(err**2))
        out[f"{prefix}mse"] = mse
        out[f"{prefix}rmse"] = float(np.sqrt(mse))
        out[f"{prefix}mae"] = float(np.mean(np.abs(err)))
        out[f"{prefix}mape"] = float(100.0 * np.mean(np.abs(err) / np.abs(t)))
        out[f"{prefix}bias"] = float(np.mean(err))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        out[f"{prefix}r2"] = float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan")

    for j, name in enumerate(names):
        _block(y_true_c[:, j], y_pred_c[:, j], f"{name}/")
    _block(y_true_c.ravel(), y_pred_c.ravel(), "")
    # Overall R^2 = mean of the per-target R^2 (pooling targets would count the
    # differences between occupants as explained variance).
    out["r2"] = float(np.nanmean([out[f"{name}/r2"] for name in names]))
    return out
