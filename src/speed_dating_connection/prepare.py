"""Collapse duplicated directed records into symmetric encounter-level dyads."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = {
    "wave",
    "iid",
    "pid",
    "match",
    "int_corr",
    "samerace",
    "age",
    "age_o",
    "attr",
    "attr_o",
    "sinc",
    "sinc_o",
    "intel",
    "intel_o",
    "fun",
    "fun_o",
    "amb",
    "amb_o",
    "shar",
    "shar_o",
    "like",
    "like_o",
    "prob",
    "prob_o",
}
BANNED_PREDICTORS = {"dec", "dec_o", "match_es", "you_call", "them_cal", "date_3"}


def _read_source(path: Path) -> Any:
    import pandas as pd

    if path.suffix.lower() != ".zip":
        return pd.read_csv(path, encoding="latin-1")
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {candidates}")
        with archive.open(candidates[0]) as handle:
            return pd.read_csv(handle, encoding="latin-1")


def _balance(left: Any, right: Any) -> Any:
    import numpy as np

    maximum = np.maximum(left, right)
    result = np.ones_like(maximum, dtype=float)
    return np.divide(np.minimum(left, right), maximum, out=result, where=maximum != 0)


def _mean(left: Any, right: Any) -> Any:
    return (left + right) / 2.0


def prepare_dyads(source_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Create one canonical, symmetric record per speed-date dyad."""
    import pandas as pd

    source = Path(source_path)
    frame = _read_source(source).copy()
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not BANNED_PREDICTORS <= set(frame.columns):
        raise ValueError(
            "Expected decision/follow-up columns are absent; verify the source version"
        )
    frame = frame.assign(
        person_low=frame[["iid", "pid"]].min(axis=1),
        person_high=frame[["iid", "pid"]].max(axis=1),
    )
    keys = ["wave", "person_low", "person_high"]
    inconsistent = frame.groupby(keys)["match"].nunique().gt(1)
    if inconsistent.any():
        raise ValueError("At least one duplicated dyad has inconsistent match labels")
    dyads = frame.drop_duplicates(keys).copy()
    output = pd.DataFrame(
        {
            "wave": dyads["wave"].astype(int),
            "person_low": dyads["person_low"].astype(int),
            "person_high": dyads["person_high"].astype(int),
            "label": dyads["match"].astype(int),
            "interest_correlation": dyads["int_corr"],
            "same_race": dyads["samerace"],
            "age_gap": (dyads["age"] - dyads["age_o"]).abs(),
            "shared_interest_mean": _mean(dyads["shar"], dyads["shar_o"]),
            "shared_interest_balance": _balance(dyads["shar"], dyads["shar_o"]),
            "like_mean": _mean(dyads["like"], dyads["like_o"]),
            "like_balance": _balance(dyads["like"], dyads["like_o"]),
            "expected_reciprocation_mean": _mean(dyads["prob"], dyads["prob_o"]),
            "expected_reciprocation_balance": _balance(dyads["prob"], dyads["prob_o"]),
        }
    )
    for source_name, clean_name in (
        ("attr", "attractiveness"),
        ("sinc", "sincerity"),
        ("intel", "intelligence"),
        ("fun", "fun"),
        ("amb", "ambition"),
    ):
        output[f"{clean_name}_mean"] = _mean(
            dyads[source_name], dyads[f"{source_name}_o"]
        )
        output[f"{clean_name}_balance"] = _balance(
            dyads[source_name], dyads[f"{source_name}_o"]
        )
    output = output.replace([math.inf, -math.inf], math.nan)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    summary = {
        "source_rows": len(frame),
        "dyads": len(output),
        "positive_dyads": int(output["label"].sum()),
        "positive_rate": float(output["label"].mean()),
        "waves": int(output["wave"].nunique()),
        "singleton_directed_records": int(frame.groupby(keys).size().eq(1).sum()),
        "banned_predictors_in_output": sorted(BANNED_PREDICTORS & set(output.columns)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary
