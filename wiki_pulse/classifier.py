"""The news classifier: is a burst of edits a story breaking, or a dispute or routine work?

A pipeline step (pipelines.yaml: news_classifier) fits a logistic regression on
the bursts Claude has labeled (raw.trends, claude_label) and writes its weights
as a table, gold.news_classifier, with the threshold at which it calls news.
The burst detector reads that table and scores every burst with it, including
any Claude doesn't read: the model distills Claude's judgment into arithmetic
that's free to run on every burst.

The features, the regularization and the threshold rule come from the
classifier study (analysis/), which compared this model with XGBoost under
cross-validation grouped by article.
"""

import hashlib
import math
import re

import numpy as np
import pyarrow as pa

INTERCEPT = "(intercept)"
YEAR = re.compile(r"(?:19|20)\d{2}")  # 2026 World Series, Manifestations du 17 octobre 2026
# Each feature from a burst's numbers, its title and its wiki's size tier. Counts are logged: 40 edits isn't 10x
# as telling as 4. The usual count stands in for surprise, which is (edits - usual) / spread: surprise moves with
# edits (correlation 0.93), so their weights blew up in opposite directions; edits and the usual count barely
# correlate, predict as well, and give weights that read as effects.
FEATURES = {
    "log_edits": lambda b: math.log1p(b["edits_30m"]),
    "log_editors": lambda b: math.log1p(b["editors_30m"]),
    "log_expected": lambda b: math.log(max(b["expected_edits"], 0.01)),
    "bot_share": lambda b: b["bot_share"],
    "anonymous_share": lambda b: b["anonymous_share"],
    "revert_share": lambda b: b["revert_share"],
    "large_wiki": lambda b: 1.0 if b.get("size_tier") == "large" else 0.0,
    "year_in_title": lambda b: 1.0 if YEAR.search(b["title"] or "") else 0.0,
}
MIN_LABELS, MIN_EACH = 30, 5  # labeled bursts needed before training, and of each kind
FOLDS = 5  # for the out-of-fold probabilities that choose the threshold and judge the model
TARGET_PRECISION = 0.5  # of the bursts called news, the share that should be news


def features(burst: dict) -> list[float]:
    return [feature(burst) for feature in FEATURES.values()]


def fit(context) -> pa.Table | None:
    """The step: weights fitted on every labeled burst so far, or None until there are enough."""
    labeled = context.query(
        "SELECT t.wiki, t.title, t.edits_30m, t.editors_30m, t.expected_edits, t.bot_share, t.anonymous_share, "
        "t.revert_share, coalesce(s.size_tier, 'unknown') AS size_tier, t.claude_label "
        "FROM raw.trends AS t LEFT JOIN gold.wiki_sizes AS s ON s.wiki = t.wiki "
        "WHERE t.claude_label IS NOT NULL AND t.claude_label <> 'unclear'"
    ).to_pylist()
    y = np.array([1.0 if row["claude_label"] == "news" else 0.0 for row in labeled])
    news = int(y.sum())
    if len(labeled) < MIN_LABELS or min(news, len(labeled) - news) < MIN_EACH:
        context.log(f"news_classifier: {len(labeled)} labeled bursts, {news} of them news; training waits for "
                    f"{MIN_LABELS}, with at least {MIN_EACH} of each kind")
        return None
    x = np.array([features(row) for row in labeled])
    # Judge the model, and choose its threshold, on probabilities from models that never saw the burst's article
    unseen = out_of_fold(x, y, [f"{row['wiki']}|{row['title']}" for row in labeled])
    threshold, precision, recall = choose_threshold(unseen, y)
    weights, center, scale = fit_weights(x, y)
    accuracy = float(((predict_many(weights, (x - center) / scale) >= threshold) == (y == 1.0)).mean())
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
        "threshold": [threshold] * count,
        "threshold_precision": [precision] * count,
        "threshold_recall": [recall] * count,
        "cv_average_precision": [average_precision(unseen, y)] * count,
        "trained_at": [context.now] * count,
    })


def fit_weights(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights fitted on standardized features, and each feature's center and scale."""
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    return train((x - center) / scale, y), center, scale


def train(z: np.ndarray, y: np.ndarray, l2: float = 1.0, iterations: int = 50) -> np.ndarray:
    """Logistic regression by Newton's method, with an L2 penalty (not on the intercept) to keep
    weights sane on a few hundred labels. Returns [intercept, weight per feature]. The study tuned
    the penalty in nested cross-validation and chose 1 (scikit-learn's C = 1)."""
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


def folds(articles: list[str], y: np.ndarray, k: int = FOLDS) -> np.ndarray:
    """Each burst's fold. An article flagged several times keeps all its flags in one fold, so a model is never
    judged on a story it trained on; articles with news are dealt out first, so every fold gets some. The order
    is a hash of the article, so retraining on the same labels gives the same folds."""
    news = {a for a, label in zip(articles, y) if label}
    order = sorted(set(articles), key=lambda a: (a not in news, hashlib.md5(a.encode()).hexdigest()))
    fold_of = {a: i % k for i, a in enumerate(order)}
    return np.array([fold_of[a] for a in articles])


def out_of_fold(x: np.ndarray, y: np.ndarray, articles: list[str]) -> np.ndarray:
    """Each burst's probability from a model fitted on the other folds."""
    fold, p = folds(articles, y), np.zeros(len(y))
    for k in range(FOLDS):
        test = fold == k
        if not test.any():
            continue
        if len(set(y[~test])) < 2:  # a fold whose rest is all one label can't fit a model: it gets the base rate
            p[test] = y[~test].mean()
            continue
        weights, center, scale = fit_weights(x[~test], y[~test])
        p[test] = predict_many(weights, (x[test] - center) / scale)
    return p


def precision_recall(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall when calling news at each distinct probability, from the highest down."""
    order = np.argsort(-p, kind="stable")
    ranked, labels = p[order], y[order]
    tp, fp = np.cumsum(labels), np.cumsum(1 - labels)
    last = np.r_[ranked[1:] != ranked[:-1], True]  # the last burst of each run of equal probabilities
    return ranked[last], tp[last] / (tp[last] + fp[last]), tp[last] / max(labels.sum(), 1)


def average_precision(p: np.ndarray, y: np.ndarray) -> float:
    """The area under the precision-recall curve (as scikit-learn computes it): random guessing scores the share of news."""
    _, precision, recall = precision_recall(p, y)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def choose_threshold(p: np.ndarray, y: np.ndarray, target: float = TARGET_PRECISION) -> tuple[float, float, float]:
    """The threshold that catches the most news while precision stays at `target` or more, and its precision and
    recall; if no threshold reaches the target, the one with the best F1."""
    thresholds, precision, recall = precision_recall(p, y)
    reaching = np.flatnonzero(precision >= target)
    if len(reaching):
        i = reaching[np.argmax(recall[reaching])]
    else:
        i = int(np.argmax(2 * precision * recall / np.maximum(precision + recall, 1e-12)))
    return float(thresholds[i]), float(precision[i]), float(recall[i])


def predict_many(weights: np.ndarray, z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(weights[0] + z @ weights[1:])))


def predict(model: list[dict] | None, burst: dict) -> float | None:
    """The probability that `burst` is news, from the weights table's rows; None without a model, or with one
    trained on features this code no longer computes (after an upgrade, until the step retrains)."""
    if not model or any(row["feature"] not in FEATURES and row["feature"] != INTERCEPT for row in model):
        return None
    values = dict(zip(FEATURES, features(burst)))
    z = 0.0
    for row in model:
        x = 1.0 if row["feature"] == INTERCEPT else (values[row["feature"]] - row["center"]) / row["scale"]
        z += row["weight"] * x
    return round(1.0 / (1.0 + math.exp(-z)), 4)
