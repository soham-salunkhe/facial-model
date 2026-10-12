"""
training/train_classifier.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standalone training script for the InterviewClassifier.

Run this script directly to regenerate the cached model:
    python training/train_classifier.py

Features
--------
- Generates synthetic training data via synthetic_data.py.
- Trains XGBoost + isotonic calibration.
- Reports cross-validation accuracy and per-class metrics.
- Saves model + scaler to models/.
"""

from __future__ import annotations

import os
import sys
import time
import pickle

# Add project root to path so config and modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

import config as cfg
from training.synthetic_data import generate


def main() -> None:
    print("=" * 60)
    print("  Interview State Classifier — Training Script")
    print("=" * 60)

    # ── Generate data ────────────────────────────────────────────────
    t0 = time.time()
    print("\n[1/4] Generating synthetic training data…")
    X, y = generate(n_samples=8000, seed=42)
    print(f"      {X.shape[0]} samples × {X.shape[1]} features")
    dist = {cfg.STATE_LABELS[i]: int((y == i).sum()) for i in range(5)}
    print(f"      Class distribution: {dist}")

    # ── Scale features ───────────────────────────────────────────────
    print("\n[2/4] Scaling features…")
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Cross-validation ─────────────────────────────────────────────
    print("\n[3/4] Cross-validating (3-fold StratifiedKFold)…")
    base = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(base, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"      CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Final training with calibration ─────────────────────────────
    print("\n[4/4] Training final model with isotonic calibration…")
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_scaled, y)

    # ── Evaluation on full training set ─────────────────────────────
    y_pred = model.predict(X_scaled)
    print("\n  Classification Report (training set):")
    print(
        classification_report(
            y, y_pred,
            target_names=cfg.STATE_LABELS,
            digits=3,
        )
    )

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    with open(cfg.CLASSIFIER_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(cfg.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    elapsed = time.time() - t0
    print(f"\n✓ Model saved → {cfg.CLASSIFIER_PATH}")
    print(f"✓ Scaler saved → {cfg.SCALER_PATH}")
    print(f"✓ Total training time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
