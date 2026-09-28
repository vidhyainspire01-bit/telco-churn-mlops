"""Promotion gate: pure functions, no Databricks/MLflow imports, so CI can unit-test them."""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def evaluate(model, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred),
    }


@dataclass
class GateConfig:
    min_pr_auc: float = 0.60        # absolute floor: never ship below this, even with no champion
    min_recall: float = 0.65        # business guardrail at the model's own threshold
    min_improvement: float = 0.005  # challenger must beat champion PR-AUC by this much
                                    # (smaller gaps are noise on ~1,450 test rows; avoids flip-flopping)


def decide(challenger: dict, champion: Optional[dict], cfg: GateConfig) -> tuple[bool, str]:
    """Return (promote, reason). `champion` is None when nothing is deployed yet."""
    if challenger["pr_auc"] < cfg.min_pr_auc:
        return False, f"pr_auc {challenger['pr_auc']:.4f} below floor {cfg.min_pr_auc}"
    if challenger["recall"] < cfg.min_recall:
        return False, f"recall {challenger['recall']:.4f} below guardrail {cfg.min_recall}"
    if champion is None:
        return True, "no current champion; challenger passes absolute checks"

    gain = challenger["pr_auc"] - champion["pr_auc"]
    if gain < cfg.min_improvement:
        return False, (f"pr_auc gain {gain:+.4f} vs champion is below required "
                       f"{cfg.min_improvement} (champion {champion['pr_auc']:.4f})")
    return True, f"pr_auc gain {gain:+.4f} vs champion {champion['pr_auc']:.4f}"