import numpy as np
import pandas as pd
from functools import reduce


def build_weighted_ensemble_submission(
    files,
    scores,
    output_path="submission_weighted_vote.csv",
    n_models=None,
):
    """
    Build a weighted-vote ensemble submission from multiple prediction files.

    Parameters
    ----------
    files : list of str
        Paths to prediction CSV files. Each file must contain:
        - sample_index
        - label
    scores : array-like
        Quality scores associated with each prediction file
        (e.g. validation F1 scores), in the same order as `files`.
    output_path : str, default="submission_weighted_vote.csv"
        Path where the final submission file will be saved.
    n_models : int or None, default=None
        Number of models to include. If None, all models are used.

    Returns
    -------
    final_submission : pd.DataFrame
        Final weighted-vote submission dataframe.
    weights : np.ndarray
        Normalized weights used for voting.
    """
    if len(files) == 0:
        raise ValueError("`files` must contain at least one prediction file.")

    scores = np.asarray(scores, dtype=float)

    if len(files) != len(scores):
        raise ValueError("`files` and `scores` must have the same length.")

    if n_models is None:
        n_models = len(files)

    files = files[:n_models]
    scores = scores[:n_models]

    if scores.sum() == 0:
        raise ValueError("The sum of the provided scores must be greater than zero.")

    weights = scores / scores.sum()

    dfs = []
    for i, path in enumerate(files, start=1):
        df = pd.read_csv(path)

        required_cols = {"sample_index", "label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{path} must contain columns {required_cols}.")

        df = df.rename(columns={"label": f"label_{i}"})
        dfs.append(df)

    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on="sample_index", how="inner"),
        dfs,
    )

    def weighted_vote(row):
        label_scores = {}
        for i, weight in enumerate(weights, start=1):
            label = row[f"label_{i}"]
            label_scores[label] = label_scores.get(label, 0.0) + weight
        return max(label_scores.items(), key=lambda x: x[1])[0]

    df_merged["label"] = df_merged.apply(weighted_vote, axis=1)

    final_submission = df_merged[["sample_index", "label"]].copy()
    final_submission.to_csv(output_path, index=False)

    return final_submission, weights
