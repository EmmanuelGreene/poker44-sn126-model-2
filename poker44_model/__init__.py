"""Poker44 model-2 package.

Model-2 keeps the same feature extractor and serving contract as the baseline model:
validator-sanitized live chunks -> 180 C2 behavioral features -> blended
sklearn probabilities -> within-batch rank-normalized bot-risk scores.
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
