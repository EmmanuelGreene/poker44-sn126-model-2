"""Poker44 model-2 blended detector.

Serving recipe: 80% primary C2 ensemble + 20% secondary C2 ensemble, using the same 180-feature C2 sanitized behavioral feature set and
within-batch rank-normalized output as the baseline model.

Both component artifacts are public in this repo as model_a.joblib and model_b.joblib.

Live validator chunks are already sanitized by the validator. Do not sanitize
again at inference.
"""
from __future__ import annotations

import os

import joblib
import numpy as np

from poker44_model.features import FEATURE_NAMES, chunk_features

MODEL_A_WEIGHT = float(os.getenv("POKER44_MODEL_A_BLEND_WEIGHT", "0.80"))
MODEL_B_WEIGHT = 1.0 - MODEL_A_WEIGHT

_MODEL_A = None
_MODEL_B = None


def _load_model(filename: str):
    return joblib.load(os.path.join(os.path.dirname(__file__), filename))


def _model_a():
    global _MODEL_A
    if _MODEL_A is None:
        _MODEL_A = _load_model("model_a.joblib")
    return _MODEL_A


def _model_b():
    global _MODEL_B
    if _MODEL_B is None:
        _MODEL_B = _load_model("model_b.joblib")
    return _MODEL_B


def _rank_normalize(vals):
    n = len(vals)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = round(pos / (n - 1), 6)
    return out


def _feature_matrix(chunks):
    rows = []
    for chunk in chunks:
        feats = chunk_features(chunk)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.array(rows, dtype=float)


def _raw_scores(chunks):
    chunks = chunks or []
    if not chunks:
        return np.array([], dtype=float)
    X = _feature_matrix(chunks)
    pa = _model_a().predict_proba(X)[:, 1]
    pb = _model_b().predict_proba(X)[:, 1]
    return MODEL_A_WEIGHT * pa + MODEL_B_WEIGHT * pb


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk, ranked within the batch."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        return _rank_normalize(list(_raw_scores(chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk blended probability fallback."""
    try:
        if not chunk:
            return 0.5
        return round(float(_raw_scores([chunk])[0]), 6)
    except Exception:
        return 0.5
