"""The news classifier study: which features and which model best tell news from other bursts, judged honestly.

Run `python features.py` first (it reads the lakehouse), then this. It writes out/results.json, which
the report page renders. Every comparison uses cross-validation grouped by article, because an article
that keeps growing is flagged, and labeled, several times: a random split would train on one flag of
a story and test on the next. Hyperparameters are tuned inside each training fold (nested CV), so the
scores are out-of-sample for the tuning too.
"""

import json
import pickle
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import load

sys.path.insert(0, str(Path(__file__).parents[1]))
from wiki_pulse import classifier  # noqa: E402  the deployed fit, to check it against scikit-learn's

OUT = Path(__file__).parent / "out" / "results.json"
CV_CACHE = Path(__file__).parent / "out" / "cv_runs.pkl"  # the nested CV's out-of-fold scores, the slow part
REPEATS = 10  # repetitions of 5-fold CV, each with its own shuffle of articles into folds
BOOTSTRAP = 1000  # resamples of articles, for confidence intervals
INCIDENT = (pd.Timestamp("2026-10-16 02:00"), pd.Timestamp("2026-10-17 16:20"))  # baselines inflated by duplicates
TIME_SPLIT = pd.Timestamp("2026-10-17 12:00")

FEATURES = {  # name: (group, description)
    "log_edits": ("size", "Edits in the 30 minutes, logged"),
    "log_editors": ("size", "Different editors in the 30 minutes, logged"),
    "log_surprise": ("size", "Standard deviations above usual, logged"),
    "log_expected": ("baseline", "What's usual in 30 minutes for an article on that wiki at that hour, logged"),
    "bot_share": ("who", "Share of edits by bots"),
    "anonymous_share": ("who", "Share of edits by editors without an account"),
    "revert_share": ("reverts", "Share of edits undoing others, English summaries only"),
    "revert_share_any_language": ("reverts", "Share of edits undoing others, in 20 languages' summaries"),
    "year_in_title": ("title", "The title holds a year, e.g. 2026 World Series"),
    "log_prior_24h": ("history", "The article's edits in the 24 hours before the window, logged"),
    "log_earlier_flags": ("history", "Earlier flags of the article in the last 24 hours, logged"),
    "log_bytes_removed": ("content", "Bytes the window's edits removed, logged"),
    "new_page": ("content", "The article was created within the window"),
    "span_minutes": ("pace", "Minutes from the window's first edit to its last"),
    "large_wiki": ("wiki", "On a large wiki (100,000+ edits a week)"),
}
DEPLOYED = ["log_edits", "log_editors", "log_surprise", "bot_share", "anonymous_share", "revert_share"]
ALL = list(FEATURES)
CANDIDATES = [f for f in ALL if f not in DEPLOYED]
# v2: the deployed six plus the two candidates that each raised average precision when added alone, with
# surprise swapped for the baseline it's measured against: surprise is (edits - usual) / spread, so it moves
# with edits (correlation 0.93) and the two weights blow up in opposite directions; edits and the baseline are
# nearly uncorrelated (0.04), score the same, and give weights that read as effects. large_wiki and
# year_in_title were picked on this data, so lr_v2's score is optimistic; lr_selected repeats the picking inside
# every training fold, which scores the procedure honestly.
V2 = ["log_edits", "log_editors", "log_expected", "bot_share", "anonymous_share", "revert_share", "large_wiki",
      "year_in_title"]  # fmt: skip
SELECT = "forward selection"

LR_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]
XGB_GRID = [
    dict(max_depth=d, n_estimators=n, learning_rate=lr, min_child_weight=w)
    for d, n, lr, w in product([1, 2, 3], [100, 300], [0.03, 0.1], [1, 5])
]
XGB_DEFAULTS = dict(max_depth=6, n_estimators=100, learning_rate=0.3, min_child_weight=1)  # out of the box


def prepare(bursts: pd.DataFrame) -> pd.DataFrame:
    d = bursts.copy()
    d["event_ts"] = pd.to_datetime(d["event_ts"])
    d["log_edits"] = np.log1p(d.edits_30m)
    d["log_editors"] = np.log1p(d.editors_30m)
    d["log_surprise"] = np.log1p(d.surprise.clip(lower=0))
    d["log_expected"] = np.log(d.expected_edits)
    d["year_in_title"] = d.title.str.contains(r"(?:19|20)\d{2}").astype(float)
    d["log_prior_24h"] = np.log1p(d.prior_24h)
    d["log_earlier_flags"] = np.log1p(d.earlier_flags_24h)
    d["log_bytes_removed"] = np.log1p(d.bytes_removed)
    d["new_page"] = d.new_page.astype(float)
    d["large_wiki"] = (d.size_tier == "large").astype(float)
    d["article"] = d.wiki + "|" + d.title
    d["labeled"] = d.claude_label.isin(["news", "edit war", "routine"])
    d["y"] = (d.claude_label == "news").astype(int)
    d["in_incident"] = d.event_ts.between(*INCIDENT)
    return d


# Models: each `make` takes hyperparameters and returns an unfitted scikit-learn estimator.
def make_lr(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))


def make_xgb(params: dict):
    return XGBClassifier(**params, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                         tree_method="hist", n_jobs=4, random_state=0, verbosity=0)  # fmt: skip


MODELS = {  # key: (label, features, make, grid of hyperparameters, or SELECT)
    "lr_deployed": ("Logistic regression, deployed features (C=1, as live)", DEPLOYED, make_lr, [1.0]),
    "lr_v2": ("Logistic regression, v2 features (C tuned)", V2, make_lr, LR_GRID),
    "lr_selected": ("Logistic regression, features selected in each fold", ALL, make_lr, SELECT),
    "lr_all": ("Logistic regression, all 14 features (C tuned)", ALL, make_lr, LR_GRID),
    "xgb_deployed": ("XGBoost, deployed features (tuned)", DEPLOYED, make_xgb, XGB_GRID),
    "xgb_v2": ("XGBoost, v2 features (tuned)", V2, make_xgb, XGB_GRID),
    "xgb_all": ("XGBoost, all 14 features (tuned)", ALL, make_xgb, XGB_GRID),
    "xgb_all_defaults": ("XGBoost, all 14 features (library defaults)", ALL, make_xgb, [XGB_DEFAULTS]),
}


def folds(y, groups, seed: int, n: int = 5):
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def tune(make, grid, x, y, groups, seed: int):
    """The grid point with the best average precision over inner folds of the training data."""
    if len(grid) == 1:
        return grid[0]
    inner = folds(y, groups, seed, n=4)
    scores = []
    for params in grid:
        oof = np.zeros(len(y))
        for train, test in inner:
            oof[test] = make(params).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
        scores.append(average_precision_score(y, oof))
    return grid[int(np.argmax(scores))]


def metrics(y, p, weights=None) -> dict:
    top = max(1, round(len(y) * 0.1))
    order = np.argsort(-p)
    return {
        "ap": average_precision_score(y, p, sample_weight=weights),
        "roc_auc": roc_auc_score(y, p, sample_weight=weights),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), sample_weight=weights),
        "brier": brier_score_loss(y, p, sample_weight=weights),
        "precision_top10": float(y[order[:top]].mean()),
    }


def forward_select(x, cols: list[str], y, groups, seed: int) -> dict:
    """Start from the deployed features and add, one at a time, the candidate that most raises average precision
    over inner folds, while it raises it by 0.01 or more; then tune C for the chosen set."""
    inner = folds(y, groups, seed, n=4)

    def score(idx, c):
        oof = np.zeros(len(y))
        for train, test in inner:
            oof[test] = make_lr(c).fit(x[np.ix_(train, idx)], y[train]).predict_proba(x[np.ix_(test, idx)])[:, 1]
        return average_precision_score(y, oof)

    chosen = [cols.index(f) for f in DEPLOYED]
    best = score(chosen, 3.0)
    while trials := {i: score(chosen + [i], 3.0) for i in range(len(cols)) if i not in chosen}:
        i, s = max(trials.items(), key=lambda kv: kv[1])
        if s < best + 0.01:
            break
        chosen.append(i)
        best = s
    return {"features": [cols[i] for i in chosen], "c": max(LR_GRID, key=lambda c: score(chosen, c))}


def nested_cv(data: pd.DataFrame, key: str) -> dict:
    """Out-of-fold probabilities for each repeat, and the hyperparameters (or features) each outer fold chose."""
    _, cols, make, grid = MODELS[key]
    x, y, groups = data[cols].to_numpy(float), data.y.to_numpy(), data.article.to_numpy()
    oof, chosen = np.zeros((REPEATS, len(y))), []
    for r in range(REPEATS):
        for k, (train, test) in enumerate(folds(y, groups, seed=r)):
            if grid == SELECT:
                params = forward_select(x[train], cols, y[train], groups[train], seed=100 * r + k)
                idx = [cols.index(f) for f in params["features"]]
                model = make_lr(params["c"]).fit(x[np.ix_(train, idx)], y[train])
                oof[r, test] = model.predict_proba(x[np.ix_(test, idx)])[:, 1]
            else:
                params = tune(make, grid, x[train], y[train], groups[train], seed=100 * r + k)
                oof[r, test] = make(params).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
            chosen.append(params)
    return {"oof": oof, "chosen": chosen}


def grouped_bootstrap(y, groups, scores: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    """Average precision of each score on BOOTSTRAP resamples of articles (the same resamples for all)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    out = {name: [] for name in scores}
    while len(next(iter(out.values()))) < BOOTSTRAP:
        rows = np.concatenate([members[g] for g in rng.choice(unique, len(unique))])
        if y[rows].sum() == 0:
            continue
        for name, p in scores.items():
            out[name].append(average_precision_score(y[rows], p[rows]))
    return {name: np.array(v) for name, v in out.items()}


def interval(values) -> list[float]:
    return [float(np.percentile(values, 5)), float(np.percentile(values, 95))]


def pr_curve(y, p) -> list[list[float]]:
    precision, recall, _ = precision_recall_curve(y, p)
    return [[round(float(r), 4), round(float(q), 4)] for r, q in zip(recall[::-1], precision[::-1])]


def calibration(y, p, edges=(0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if inside.sum():
            rows.append({"band": f"{lo:.0%}-{hi:.0%}", "predicted": float(p[inside].mean()),
                         "observed": float(y[inside].mean()), "n": int(inside.sum())})  # fmt: skip
    return rows


def learning_curve(data: pd.DataFrame, settings: dict, fractions=(0.25, 0.5, 0.75, 1.0)) -> dict:
    """Average precision on the outer test folds when training on a fraction of the training articles."""
    out = {}
    y, groups = data.y.to_numpy(), data.article.to_numpy()
    for key, (cols, make, params) in settings.items():
        x = data[cols].to_numpy(float)
        points = []
        for fraction in fractions:
            aps, news = [], []
            for r in range(REPEATS):
                rng = np.random.default_rng(1000 + r)
                oof, seen = np.zeros(len(y)), np.zeros(len(y), bool)
                for train, test in folds(y, groups, seed=r):
                    articles = np.unique(groups[train])
                    keep = set(rng.choice(articles, max(2, round(len(articles) * fraction)), replace=False))
                    subset = train[np.isin(groups[train], list(keep))]
                    if y[subset].sum() < 2:
                        continue
                    news.append(int(y[subset].sum()))
                    oof[test] = make(params).fit(x[subset], y[subset]).predict_proba(x[test])[:, 1]
                    seen[test] = True
                aps.append(average_precision_score(y[seen], oof[seen]))
            points.append({"fraction": fraction, "news_in_training": float(np.mean(news)),
                           "ap": float(np.mean(aps)), "ap_interval": interval(aps)})  # fmt: skip
        out[key] = points
    return out


def ablation(data: pd.DataFrame, c: float) -> list[dict]:
    """v2 logistic regression without each feature group, against v2 with all, on the same folds."""
    y, groups = data.y.to_numpy(), data.article.to_numpy()

    def cv_ap(cols):
        x = data[cols].to_numpy(float)
        aps = []
        for r in range(REPEATS):
            oof = np.zeros(len(y))
            for train, test in folds(y, groups, seed=r):
                oof[test] = make_lr(c).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
            aps.append(average_precision_score(y, oof))
        return np.array(aps)

    full = cv_ap(V2)
    rows = [{"kind": "v2", "change": "v2 as it is", "features": V2, "ap": float(full.mean()), "delta": 0.0}]

    def row(kind, change, features, cols):
        aps = cv_ap(cols)
        rows.append({"kind": kind, "change": change, "features": features, "ap": float(aps.mean()),
                     "delta": float((aps - full).mean()), "delta_interval": interval(aps - full),
                     "ahead_share": float((aps > full).mean())})  # fmt: skip

    for group in dict.fromkeys(FEATURES[f][0] for f in V2):
        dropped = [f for f in V2 if FEATURES[f][0] == group]
        row("without", f"without {group}", dropped, [f for f in V2 if f not in dropped])
    for extra in [f for f in CANDIDATES if f not in V2 and f != "revert_share_any_language"]:
        row("with", f"with {extra}", [extra], V2 + [extra])
    row("with", "with reverts in 20 languages", ["revert_share_any_language"],
        ["revert_share_any_language" if f == "revert_share" else f for f in V2])
    return rows


def main() -> None:
    started = time.time()
    everything = prepare(load())
    data = everything[everything.labeled].reset_index(drop=True)
    y, groups = data.y.to_numpy(), data.article.to_numpy()
    results: dict = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "repeats": REPEATS,
                     "bootstrap": BOOTSTRAP}  # fmt: skip

    # The data
    flagged = everything.assign(day=everything.event_ts.dt.date.astype(str), hour=everything.event_ts.dt.hour)
    results["data"] = {
        "bursts": len(everything), "articles": int(everything.article.nunique()),
        "first": str(everything.event_ts.min()), "last": str(everything.event_ts.max()),
        "labels": {k: int(v) for k, v in everything.claude_label.fillna("not read").value_counts().items()},
        "labeled": len(data), "news": int(y.sum()), "news_articles": int(data[data.y == 1].article.nunique()),
        "labeled_articles": int(data.article.nunique()),
        "coverage_by_hour": [
            {"hour": int(h), "flagged": int(len(g)), "labeled": int(g.labeled.sum())}
            for h, g in flagged[flagged.event_ts >= "2026-10-14"].groupby("hour")
        ],
        "by_tier": {t: {k: int(v) for k, v in g.claude_label.value_counts().items()} for t, g in data.groupby("size_tier")},
    }  # fmt: skip
    results["features"] = {
        name: {"group": group, "description": text, "in_v2": name in V2, "in_deployed": name in DEPLOYED,
               "median": {label: float(g[name].median()) for label, g in data.groupby("claude_label")},
               "mean": {label: float(g[name].mean()) for label, g in data.groupby("claude_label")}}
        for name, (group, text) in FEATURES.items()
    }  # fmt: skip

    # Newton's method (the deployed fit) against scikit-learn's solver, on the same data and penalty
    x = data[DEPLOYED].to_numpy(float)
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    newton = classifier.train(z, y.astype(float))
    lbfgs = LogisticRegression(C=1.0, max_iter=1000).fit(z, y)
    results["newton_check"] = {"features": ["(intercept)", *DEPLOYED], "newton": newton.tolist(),
                               "lbfgs": [float(lbfgs.intercept_[0]), *map(float, lbfgs.coef_[0])],
                               "max_difference": float(np.abs(newton - np.r_[lbfgs.intercept_, lbfgs.coef_[0]]).max())}  # fmt: skip
    print(f"newton vs lbfgs: max weight difference {results['newton_check']['max_difference']:.2e}")

    # The models, nested CV
    cache = pickle.loads(CV_CACHE.read_bytes()) if CV_CACHE.exists() else {}
    runs = {}
    for key, (_, cols, _, grid) in MODELS.items():
        signature = json.dumps([len(data), int(y.sum()), REPEATS, cols, grid], sort_keys=True)
        if signature not in cache:
            t = time.time()
            cache[signature] = nested_cv(data, key)
            CV_CACHE.write_bytes(pickle.dumps(cache))
            print(f"{key}: {time.time() - t:.0f}s", flush=True)
        runs[key] = cache[signature]
    live = data.p_news.notna().to_numpy()
    averaged = {key: run["oof"].mean(axis=0) for key, run in runs.items()}
    boot = grouped_bootstrap(y, groups, averaged)
    results["models"] = []
    for key, (label, cols, _, grid) in MODELS.items():
        per_repeat = [metrics(y, p) for p in runs[key]["oof"]]
        summary = {m: float(np.mean([r[m] for r in per_repeat])) for m in per_repeat[0]}
        chosen = Counter(json.dumps(c, sort_keys=True) for c in runs[key]["chosen"]).most_common(3)
        results["models"].append({
            "key": key, "label": label, "features": cols, "metrics": summary,
            "ap_repeats": [r["ap"] for r in per_repeat], "ap_bootstrap_interval": interval(boot[key]),
            "chosen": [{"params": json.loads(c), "folds": n} for c, n in chosen],
        })  # fmt: skip
    results["base_rate"] = float(y.mean())
    pairs = [("lr_v2", "lr_deployed"), ("lr_selected", "lr_deployed"), ("lr_v2", "lr_all"), ("lr_v2", "xgb_v2"),
             ("lr_v2", "xgb_all"), ("xgb_all", "xgb_all_defaults"), ("lr_deployed", "xgb_deployed")]  # fmt: skip
    results["pairs"] = [
        {"a": a, "b": b, "delta_ap": float((boot[a] - boot[b]).mean()), "interval": interval(boot[a] - boot[b]),
         "a_better_share": float((boot[a] > boot[b]).mean()),
         "a_better_repeats": float(np.mean([ra > rb for ra, rb in zip(*(next(m["ap_repeats"] for m in results["models"] if m["key"] == k) for k in (a, b)))]))}
        for a, b in pairs
    ]  # fmt: skip
    results["pr_curves"] = {key: pr_curve(y, averaged[key]) for key in ("lr_deployed", "lr_v2", "xgb_v2", "xgb_all")}
    results["calibration"] = {key: calibration(y, averaged[key]) for key in ("lr_v2", "xgb_v2", "xgb_all")}
    picks = [c for c in runs["lr_selected"]["chosen"]]
    results["selection"] = {
        "folds": len(picks),
        "picked": {f: float(np.mean([f in c["features"] for c in picks])) for f in CANDIDATES},
        "c": {str(k): v for k, v in Counter(c["c"] for c in picks).items()},
        "sets": [{"features": json.loads(k), "folds": v} for k, v in
                 Counter(json.dumps([f for f in c["features"] if f not in DEPLOYED]) for c in picks).most_common(5)],
    }  # fmt: skip
    results["calibration"]["live"] = calibration(y[live], data.p_news.to_numpy()[live])
    results["live"] = {"n": int(live.sum()), "news": int(y[live].sum()), **metrics(y[live], data.p_news.to_numpy()[live])}

    # Settle the tuned hyperparameters (the most chosen) for what follows
    c_best = Counter(c for c in runs["lr_v2"]["chosen"]).most_common(1)[0][0]
    settle = lambda key: json.loads(Counter(json.dumps(c, sort_keys=True) for c in runs[key]["chosen"]).most_common(1)[0][0])  # noqa: E731
    xgb_best, xgb_all_best = settle("xgb_v2"), settle("xgb_all")
    results["settled"] = {"lr_c": c_best, "xgb": xgb_best, "xgb_all": xgb_all_best}

    # Learning curves
    t = time.time()
    results["learning_curve"] = learning_curve(data, {
        "lr_v2": (V2, make_lr, c_best), "xgb_v2": (V2, make_xgb, xgb_best), "xgb_all": (ALL, make_xgb, xgb_all_best),
    })  # fmt: skip
    print(f"learning curves: {time.time() - t:.0f}s")

    # Which features carry it
    results["ablation"] = ablation(data, c_best)
    final = make_lr(c_best).fit(data[V2].to_numpy(float), y)
    weights = final[-1].coef_[0]
    rng = np.random.default_rng(7)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    draws = []
    for _ in range(300):
        rows = np.concatenate([members[g] for g in rng.choice(unique, len(unique))])
        draws.append(make_lr(c_best).fit(data[V2].to_numpy(float)[rows], y[rows])[-1].coef_[0])
    draws = np.array(draws)
    results["coefficients"] = [
        {"feature": f, "weight": float(w), "interval": interval(draws[:, i])} for i, (f, w) in enumerate(zip(V2, weights))
    ]
    results["intercept"] = float(final[-1].intercept_[0])
    booster = make_xgb(xgb_all_best).fit(data[ALL].to_numpy(float), y)
    from xgboost import DMatrix

    contributions = booster.get_booster().predict(DMatrix(data[ALL].to_numpy(float)), pred_contribs=True)[:, :-1]
    results["shap"] = [{"feature": f, "mean_abs": float(v)} for f, v in zip(ALL, np.abs(contributions).mean(axis=0))]

    # A threshold for calling news, from v2's out-of-fold probabilities
    p = averaged["lr_v2"]
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    candidates = []
    for target in (0.4, 0.5, 0.6):
        ok = np.flatnonzero(precision[:-1] >= target)
        if len(ok):
            i = ok[np.argmax(recall[ok])]
            candidates.append({"rule": f"precision of {target:.0%} or more, most recall", "threshold": float(thresholds[i]),
                               "precision": float(precision[i]), "recall": float(recall[i])})  # fmt: skip
    i = int(np.argmax(f1[:-1]))
    candidates.append({"rule": "best F1", "threshold": float(thresholds[i]), "precision": float(precision[i]), "recall": float(recall[i])})
    scored_all = final.predict_proba(everything[V2].to_numpy(float))[:, 1]
    days = (everything.event_ts.max() - everything.event_ts.min()).total_seconds() / 86400
    for c in candidates:
        c["news_calls_per_day"] = float((scored_all >= c["threshold"]).sum() / days)
    results["thresholds"] = candidates
    results["score_distribution"] = {
        label: np.histogram(p[(data.claude_label == label).to_numpy()], bins=np.linspace(0, 1, 21))[0].tolist()
        for label in ("news", "edit war", "routine")
    }

    # Forward in time: train before the split, test after; the live model's own scores on the same bursts
    before, after = data[data.event_ts < TIME_SPLIT], data[data.event_ts >= TIME_SPLIT]
    forward = {"split": str(TIME_SPLIT), "train": len(before), "train_news": int(before.y.sum()), "test": len(after),
               "test_news": int(after.y.sum()), "models": {}}  # fmt: skip
    for key, cols, model in [("lr_deployed", DEPLOYED, make_lr(1.0)), ("lr_v2", V2, make_lr(c_best)),
                             ("xgb_v2", V2, make_xgb(xgb_best)), ("xgb_all", ALL, make_xgb(xgb_all_best))]:  # fmt: skip
        q = model.fit(before[cols].to_numpy(float), before.y).predict_proba(after[cols].to_numpy(float))[:, 1]
        forward["models"][key] = metrics(after.y.to_numpy(), q)
    has_live = after.p_news.notna()
    forward["models"]["live"] = metrics(after.y.to_numpy()[has_live], after.p_news.to_numpy()[has_live])
    results["forward"] = forward

    # The label sample's bias: weight first-come labels by how thinly their hour was covered
    cell = flagged.groupby(["day", "hour"]).agg(n_flagged=("labeled", "size"), n_labeled=("labeled", "sum"))
    d = data.assign(day=data.event_ts.dt.date.astype(str), hour=data.event_ts.dt.hour).join(cell, on=["day", "hour"])
    w = (d.n_flagged / d.n_labeled).to_numpy()
    results["reweighted"] = {"unweighted": metrics(y, p), "weighted": metrics(y, p, w),
                             "weight_range": [float(w.min()), float(w.max())]}  # fmt: skip

    # The incident window: train with or without its bursts, test outside it
    outside = ~data.in_incident.to_numpy()
    x = data[V2].to_numpy(float)
    with_, without = [], []
    for r in range(REPEATS):
        a, b = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for train, test in folds(y, groups, seed=r):
            a[test] = make_lr(c_best).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
            keep = train[outside[train]]
            b[test] = make_lr(c_best).fit(x[keep], y[keep]).predict_proba(x[test])[:, 1]
        with_.append(average_precision_score(y[outside], a[outside]))
        without.append(average_precision_score(y[outside], b[outside]))
    results["incident"] = {"labeled_inside": int((~outside).sum()), "news_inside": int(y[~outside].sum()),
                           "ap_trained_with": float(np.mean(with_)), "ap_trained_without": float(np.mean(without))}  # fmt: skip

    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
