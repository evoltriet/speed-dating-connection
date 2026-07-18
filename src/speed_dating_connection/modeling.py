"""Grouped out-of-fold feature-set comparisons for mutual interest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROFILE_FEATURES = ["interest_correlation", "same_race", "age_gap"]
SHARED_INTEREST_FEATURES = [
    "interest_correlation",
    "shared_interest_mean",
    "shared_interest_balance",
]
RECIPROCITY_FEATURES = [
    "like_mean",
    "like_balance",
    "expected_reciprocation_mean",
    "expected_reciprocation_balance",
]
TRAIT_FEATURES = [
    feature
    for trait in ("attractiveness", "sincerity", "intelligence", "fun", "ambition")
    for feature in (f"{trait}_mean", f"{trait}_balance")
]
FEATURE_SETS = {
    "profile_similarity": PROFILE_FEATURES,
    "shared_interest": SHARED_INTEREST_FEATURES,
    "reciprocity": RECIPROCITY_FEATURES,
    "perceived_traits": TRAIT_FEATURES,
    "full": PROFILE_FEATURES
    + ["shared_interest_mean", "shared_interest_balance"]
    + RECIPROCITY_FEATURES
    + TRAIT_FEATURES,
}


def _metrics(labels: Any, scores: Any) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
    }


def _cluster_bootstrap_delta(
    frame: Any,
    left_scores: Any,
    right_scores: Any,
    *,
    iterations: int = 5_000,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    waves = np.array(sorted(frame["wave"].unique()))
    random = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {"roc_auc": [], "average_precision": []}
    for _ in range(iterations):
        sampled_waves = random.choice(waves, size=len(waves), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(frame["wave"].to_numpy() == wave) for wave in sampled_waves]
        )
        labels = frame["label"].to_numpy()[indices]
        if len(np.unique(labels)) != 2:
            continue
        deltas["roc_auc"].append(
            float(
                roc_auc_score(labels, right_scores[indices])
                - roc_auc_score(labels, left_scores[indices])
            )
        )
        deltas["average_precision"].append(
            float(
                average_precision_score(labels, right_scores[indices])
                - average_precision_score(labels, left_scores[indices])
            )
        )
    return {
        metric: {
            "iterations": len(values),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for metric, values in deltas.items()
    }


def _fit_oof(frame: Any, features: list[str]) -> tuple[Any, dict[str, float]]:
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    labels = frame["label"].to_numpy()
    scores = np.zeros(len(frame), dtype=float)
    importances = np.zeros(len(features), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in splitter.split(frame, labels, groups=frame["wave"]):
        train_labels = labels[train_index]
        positives = int(train_labels.sum())
        negatives = len(train_labels) - positives
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=350,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            scale_pos_weight=negatives / positives,
            random_state=42,
            n_jobs=4,
        )
        model.fit(frame.iloc[train_index][features], train_labels, verbose=False)
        scores[test_index] = model.predict_proba(frame.iloc[test_index][features])[:, 1]
        importances += model.feature_importances_ / splitter.n_splits
    return scores, {
        feature: float(importance)
        for feature, importance in zip(features, importances, strict=True)
    }


def _render_hero(results: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    order = ["profile_similarity", "shared_interest", "reciprocity", "perceived_traits", "full"]
    labels = ["Profile\nsimilarity", "Shared\ninterest", "Reciprocity", "Perceived\ntraits", "Full"]
    values = [results["models"][name]["metrics"]["roc_auc"] for name in order]
    colors = ["#94A3B8", "#64748B", "#2563EB", "#7C3AED", "#0F766E"]
    importance = results["models"]["full"]["gain_importance"]
    top = sorted(importance.items(), key=lambda item: item[1])[-8:]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="#F8FAFC")
    bars = axes[0].bar(labels, values, color=colors, width=0.65)
    axes[0].set_title("Held-out wave ROC-AUC", loc="left", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0.5, max(values) + 0.08)
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    axes[0].grid(axis="y", color="#E2E8F0")
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
        )
    axes[1].barh(
        [name.replace("_", " ") for name, _ in top],
        [value for _, value in top],
        color="#0F766E",
    )
    axes[1].set_title("Full-model feature importance", loc="left", fontsize=13, fontweight="bold")
    axes[1].spines[["top", "right", "left", "bottom"]].set_visible(False)
    axes[1].tick_params(axis="x", bottom=False, labelbottom=False)
    figure.suptitle(
        "Interaction perceptions predict mutual interest better than profile similarity",
        x=0.06,
        y=0.96,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
    )
    figure.text(
        0.06,
        0.90,
        "4,194 dyads · 21 event waves · decisions excluded from predictors",
        color="#475569",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0.04, 0.06, 0.98, 0.85))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor=figure.get_facecolor())
    plt.close(figure)


def run_benchmark(
    dyad_path: str | Path,
    *,
    output_path: str | Path,
    predictions_path: str | Path,
    hero_chart_path: str | Path,
) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(dyad_path)
    results: dict[str, Any] = {
        "dyads": len(frame),
        "positives": int(frame["label"].sum()),
        "waves": int(frame["wave"].nunique()),
        "evaluation": "5-fold stratified group holdout by wave",
        "banned_predictors": ["dec", "dec_o", "match_es", "you_call", "them_cal", "date_3"],
        "feature_sets": FEATURE_SETS,
        "models": {},
    }
    predictions = frame[["wave", "person_low", "person_high", "label"]].copy()
    scores_by_model: dict[str, Any] = {}
    for name, features in FEATURE_SETS.items():
        scores, importance = _fit_oof(frame, features)
        scores_by_model[name] = scores
        predictions[f"score_{name}"] = scores
        results["models"][name] = {
            "metrics": _metrics(frame["label"], scores),
            "gain_importance": importance,
        }
    for comparison in ("profile_similarity", "shared_interest", "reciprocity", "perceived_traits"):
        results[f"delta_full_vs_{comparison}"] = {
            metric: results["models"]["full"]["metrics"][metric]
            - results["models"][comparison]["metrics"][metric]
            for metric in ("roc_auc", "average_precision")
        }
    results["delta_reciprocity_vs_shared_interest"] = {
        metric: results["models"]["reciprocity"]["metrics"][metric]
        - results["models"]["shared_interest"]["metrics"][metric]
        for metric in ("roc_auc", "average_precision")
    }
    results["delta_reciprocity_vs_shared_interest"]["cluster_bootstrap_95"] = (
        _cluster_bootstrap_delta(
            frame,
            scores_by_model["shared_interest"],
            scores_by_model["reciprocity"],
        )
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prediction_file = Path(predictions_path)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_file, index=False)
    _render_hero(results, Path(hero_chart_path))
    print(json.dumps(results, indent=2, sort_keys=True))
    return results
