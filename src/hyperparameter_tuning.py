from itertools import product

import numpy as np

from src.cross_validation import run_stratified_kfold_cv


def grid_search_cv_rnn(df, param_grid, fixed_params, cv_params, verbose=True):
    """
    Run grid search over recurrent-model hyperparameters using stratified
    sample-level cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features, `sample_index`, and `label`.
    param_grid : dict
        Dictionary of hyperparameters to explore.
    fixed_params : dict
        Parameters kept fixed during the search.
    cv_params : dict
        Cross-validation settings shared across all configurations.
    verbose : bool, default=True
        Whether to print progress information.

    Returns
    -------
    results : dict
        Dictionary containing cross-validation scores for each configuration.
    best_config : dict
        Best-performing hyperparameter configuration.
    best_score : float
        Mean validation score of the best configuration.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    results = {}
    best_score = -np.inf
    best_config = None

    total = len(combinations)

    for idx, combo in enumerate(combinations, start=1):
        current_config = dict(zip(param_names, combo))
        config_str = "_".join([f"{k}_{v}" for k, v in current_config.items()])

        if verbose:
            print(f"\nConfiguration {idx}/{total} — {config_str}")

        run_params = {**fixed_params, **current_config}

        fold_losses, fold_metrics, fold_scores = run_stratified_kfold_cv(
            df=df,
            experiment_name=config_str,
            **run_params,
            **cv_params,
        )

        results[config_str] = fold_scores

        if fold_scores["mean"] > best_score:
            best_score = fold_scores["mean"]
            best_config = current_config.copy()

            if verbose:
                print(f"New best score = {best_score:.4f}")

        if verbose:
            print(f"Average F1: {fold_scores['mean']:.4f} ± {fold_scores['std']:.4f}")

    return results, best_config, best_score
