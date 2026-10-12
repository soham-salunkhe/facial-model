"""
training/synthetic_data.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates expert-rule synthetic labelled training data for the
InterviewClassifier.

No external dataset is required. Data is generated using domain knowledge
rules for each interview state, with Gaussian noise to create realistic
variation. The result is a (X, y) pair where:
  X : np.ndarray  shape (n_samples, 22)  — feature vectors
  y : np.ndarray  shape (n_samples,)     — integer labels 0–4

Label mapping (matches config.STATE_LABELS order):
  0 = Confident
  1 = Nervous
  2 = Confused
  3 = Stressed
  4 = Happy
"""

from __future__ import annotations

import numpy as np


# ── Canonical "archetype" feature vectors ──────────────────────────────────
# Each archetype is a dictionary mapping feature indices to desired values.
# Indices correspond to FeatureAggregator feature layout (see feature_aggregator.py).
#
# Index map reminder:
#  0=angry  1=disgust  2=fear  3=happy  4=neutral  5=sad  6=surprise
#  7=eye_contact  8=blink_rate  9=gaze_L  10=gaze_R  11=gaze_U  12=gaze_D
#  13=pitch  14=yaw  15=roll  16=smile  17=movement  18=mouth_open
#  19=eyebrow_raise  20=head_stable  21=jitter

_ARCHETYPES = {
    # ── 0: Confident ─────────────────────────────────────────────────────
    0: {
        "base": np.array([
            0.03, 0.01, 0.03, 0.15, 0.65, 0.05, 0.08,  # emotions
            0.80,                                         # eye contact (high)
            0.50,                                         # blink rate (normal)
            0.0, 0.0, 0.0, 0.0,                         # gaze flags (mostly center)
            0.0, 0.05, 0.0,                              # head pose (stable, slight yaw ok)
            0.35,                                         # smile (moderate)
            0.10,                                         # movement (low)
            0.10,                                         # mouth open (low)
            0.30,                                         # eyebrow (neutral)
            1.0,                                          # head stable
            0.10,                                         # jitter (low)
        ], dtype=np.float32),
        "noise": np.array([
            0.05, 0.02, 0.05, 0.10, 0.10, 0.05, 0.05,
            0.12, 0.20, 0.1, 0.1, 0.05, 0.05,
            0.05, 0.10, 0.05,
            0.15, 0.08, 0.05, 0.10, 0.0, 0.05,
        ], dtype=np.float32),
    },

    # ── 1: Nervous ───────────────────────────────────────────────────────
    1: {
        "base": np.array([
            0.08, 0.02, 0.25, 0.05, 0.45, 0.10, 0.05,  # high fear, neutral
            0.45,                                         # low eye contact
            1.50,                                         # high blink rate
            0.20, 0.15, 0.05, 0.20,                     # gaze frequently averted
            0.10, 0.15, 0.05,                            # slight head movement
            0.10,                                         # low smile
            0.40,                                         # higher movement
            0.15,                                         # mouth slightly open
            0.55,                                         # eyebrow raised (tension)
            0.60,                                         # partially stable
            0.40,                                         # higher jitter
        ], dtype=np.float32),
        "noise": np.array([
            0.06, 0.02, 0.12, 0.05, 0.12, 0.06, 0.04,
            0.15, 0.40, 0.15, 0.15, 0.08, 0.15,
            0.10, 0.15, 0.05,
            0.08, 0.15, 0.08, 0.15, 0.3, 0.15,
        ], dtype=np.float32),
    },

    # ── 2: Confused ──────────────────────────────────────────────────────
    2: {
        "base": np.array([
            0.05, 0.01, 0.08, 0.05, 0.40, 0.08, 0.33,  # surprise elevated
            0.55,                                         # moderate eye contact
            0.70,                                         # moderate blink
            0.10, 0.10, 0.15, 0.10,                     # some gaze deviation (up)
            -0.20, 0.20, 0.15,                           # head tilted/up
            0.05,                                         # low smile
            0.25,                                         # moderate movement
            0.10,                                         # mouth partially open
            0.75,                                         # raised eyebrows (confusion)
            0.50,                                         # partially stable
            0.25,                                         # moderate jitter
        ], dtype=np.float32),
        "noise": np.array([
            0.04, 0.01, 0.06, 0.05, 0.12, 0.06, 0.15,
            0.15, 0.25, 0.10, 0.10, 0.12, 0.10,
            0.10, 0.15, 0.10,
            0.05, 0.10, 0.08, 0.15, 0.3, 0.10,
        ], dtype=np.float32),
    },

    # ── 3: Stressed ──────────────────────────────────────────────────────
    3: {
        "base": np.array([
            0.20, 0.05, 0.15, 0.03, 0.30, 0.22, 0.05,  # angry + sad + fear
            0.30,                                         # low eye contact
            1.80,                                         # very high blink
            0.15, 0.10, 0.0, 0.30,                      # looking down
            0.25, 0.05, 0.05,                            # head down
            0.05,                                         # no smile
            0.50,                                         # high movement
            0.20,                                         # mouth open (sighing)
            0.40,                                         # furrowed brows
            0.40,                                         # often unstable
            0.45,                                         # high jitter
        ], dtype=np.float32),
        "noise": np.array([
            0.10, 0.04, 0.10, 0.03, 0.10, 0.10, 0.04,
            0.12, 0.40, 0.12, 0.10, 0.0, 0.15,
            0.10, 0.10, 0.05,
            0.04, 0.15, 0.08, 0.15, 0.3, 0.15,
        ], dtype=np.float32),
    },

    # ── 4: Happy ─────────────────────────────────────────────────────────
    4: {
        "base": np.array([
            0.02, 0.01, 0.02, 0.72, 0.15, 0.02, 0.06,  # very happy
            0.80,                                         # high eye contact
            0.40,                                         # normal blink
            0.0, 0.0, 0.0, 0.0,                         # centered gaze
            0.0, 0.0, 0.0,                               # stable head
            0.80,                                         # high smile
            0.15,                                         # low movement
            0.25,                                         # mouth open (smiling/talking)
            0.25,                                         # normal eyebrows
            1.0,                                          # stable
            0.10,                                         # low jitter
        ], dtype=np.float32),
        "noise": np.array([
            0.02, 0.01, 0.02, 0.12, 0.08, 0.02, 0.05,
            0.12, 0.15, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.08, 0.05,
            0.12, 0.08, 0.08, 0.10, 0.0, 0.05,
        ], dtype=np.float32),
    },
}


def generate(
    n_samples: int = 6000,
    seed: int = 42,
    samples_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate (split equally across classes).
    seed : int
        Random seed for reproducibility.
    samples_per_class : int | None
        Override per-class count (defaults to n_samples // 5).

    Returns
    -------
    X : np.ndarray  shape (n_samples, 22)
    y : np.ndarray  shape (n_samples,)  — integer labels 0–4
    """
    rng = np.random.default_rng(seed)
    n_classes = len(_ARCHETYPES)
    spc = samples_per_class or (n_samples // n_classes)

    X_parts = []
    y_parts = []

    for label, arch in _ARCHETYPES.items():
        base  = arch["base"]    # shape (22,)
        noise = arch["noise"]   # shape (22,)

        # Gaussian noise around the archetype
        samples = rng.normal(
            loc=base,
            scale=noise,
            size=(spc, len(base)),
        ).astype(np.float32)

        # Clip emotion probs to [0, 1] and renormalise
        emotion_slice = samples[:, :7]
        emotion_slice = np.clip(emotion_slice, 0.0, 1.0)
        emotion_total = emotion_slice.sum(axis=1, keepdims=True)
        emotion_total = np.where(emotion_total < 1e-6, 1.0, emotion_total)
        samples[:, :7] = emotion_slice / emotion_total

        # Clip gaze flags to [0, 1]
        samples[:, 9:13] = np.clip(samples[:, 9:13], 0.0, 1.0)

        # Clip misc features to [0, 1] / valid ranges
        samples[:, 7]  = np.clip(samples[:, 7],  0.0, 1.0)   # eye contact
        samples[:, 8]  = np.clip(samples[:, 8],  0.0, 3.0)   # blink rate
        samples[:, 13] = np.clip(samples[:, 13], -1.0, 1.0)  # pitch
        samples[:, 14] = np.clip(samples[:, 14], -1.0, 1.0)  # yaw
        samples[:, 15] = np.clip(samples[:, 15], -1.0, 1.0)  # roll
        samples[:, 16] = np.clip(samples[:, 16], 0.0, 1.0)   # smile
        samples[:, 17] = np.clip(samples[:, 17], 0.0, 3.0)   # movement
        samples[:, 18] = np.clip(samples[:, 18], 0.0, 1.0)   # mouth open
        samples[:, 19] = np.clip(samples[:, 19], 0.0, 1.0)   # eyebrow
        samples[:, 20] = np.clip(samples[:, 20], 0.0, 1.0)   # stability
        samples[:, 21] = np.clip(samples[:, 21], 0.0, 1.0)   # jitter

        X_parts.append(samples)
        y_parts.append(np.full(spc, label, dtype=np.int32))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


if __name__ == "__main__":
    X, y = generate(n_samples=1000)
    print(f"Generated X: {X.shape}, y: {y.shape}")
    print(f"Label distribution: { {i: int((y == i).sum()) for i in range(5)} }")
    print(f"Feature means:\n{X.mean(axis=0).round(3)}")
