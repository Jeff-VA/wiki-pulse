"""The news classifier: is a burst of edits a story breaking, or a dispute or routine work?

A pipeline step (pipelines.yaml: news_classifier) fits a logistic regression on
the bursts Claude has labeled (raw.trends, claude_label) and writes its weights
as a table, gold.news_classifier. The burst detector reads that table and
scores every burst with it, including the many Claude never sees: the model
distills Claude's judgment into arithmetic that's free to run on every burst.
"""

import math

import numpy as np
import pyarrow as pa

INTERCEPT = "(intercept)"
# Each feature from a burst's counts. Counts are logged: 40 edits isn't 10x as telling as 4.
FEATURES = {
    "log_edits": lambda b: math.log1p(b["edits_30m"]),
    "log_editors": lambda b: math.log1p(b["editors_30m"]),
    "log_surprise": lambda b: math.log1p(max(b["surprise"], 0.0)),
    "bot_share": lambda b: b["bot_share"],
    "anonymous_share": lambda b: b["anonymous_share"],
    "revert_share": lambda b: b["revert_share"],
}
MIN_LABELS, MIN_EACH = 30, 5  # labeled bursts needed before training, and of each kind


def features(burst: dict) -> list[float]:
    return [feature(burst) for feature in FEATURES.values()]


def fit(context) -> pa.Table | None:
    """The step: weights fitted on every labeled burst so far, or None until there are enough."""
    labeled = context.query(
        "SELECT edits_30m, editors_30m, surprise, bot_share, anonymous_share, revert_share, claude_label "
        "FROM raw.trends WHERE claude_label IS NOT NULL AND claude_label <> 'unclear'"
    ).to_pylist()
    y = np.array([1.0 if row["claude_label"] == "news" else 0.0 for row in labeled])
    news = int(y.sum())
    if len(labeled) < MIN_LABELS or min(news, len(labeled) - news) < MIN_EACH:
        context.log(f"news_classifier: {len(labeled)} labeled bursts, {news} of them news; training waits for "
                    f"{MIN_LABELS}, with at least {MIN_EACH} of each kind")
        return None
    x = np.array([features(row) for row in labeled])
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    weights = train((x - center) / scale, y)
    predicted = predict_many(weights, (x - center) / scale) >= 0.5
    accuracy = float((predicted == (y == 1.0)).mean())
    names = [INTERCEPT, *FEATURES]
    count = len(names)
    return pa.table({
        "feature": names,
        "weight": [float(w) for w in weights],
        "center": [0.0, *map(float, center)],
        "scale": [1.0, *map(float, scale)],
        "samples": [len(labeled)] * count,
        "news_share": [news / len(labeled)] * count,
        "training_accuracy": [accuracy] * count,
        "trained_at": [context.now] * count,
    })


def train(z: np.ndarray, y: np.ndarray, l2: float = 1.0, iterations: int = 50) -> np.ndarray:
    """Logistic regression by Newton's method, with an L2 penalty (not on the intercept) to keep
    weights sane on a few hundred labels. Returns [intercept, weight per feature]."""
    x = np.hstack([np.ones((len(z), 1)), z])
    weights = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1]) * l2
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-x @ weights))
        gradient = x.T @ (p - y) + penalty @ weights
        hessian = x.T @ (x * (p * (1 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian + 1e-9 * np.eye(len(weights)), gradient)
        weights -= step
        if np.abs(step).max() < 1e-8:
            break
    return weights


def predict_many(weights: np.ndarray, z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(weights[0] + z @ weights[1:])))


def predict(model: list[dict] | None, burst: dict) -> float | None:
    """The probability that `burst` is news, from the weights table's rows; None without a model."""
    if not model:
        return None
    values = dict(zip(FEATURES, features(burst)))
    z = 0.0
    for row in model:
        x = 1.0 if row["feature"] == INTERCEPT else (values[row["feature"]] - row["center"]) / row["scale"]
        z += row["weight"] * x
    return round(1.0 / (1.0 + math.exp(-z)), 4)
