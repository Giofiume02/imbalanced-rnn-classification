import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

def analyze_autocorrelation(X, sample_indices, max_lag=100, n_samples=5):
    """
    Analyze autocorrelation to determine optimal window size

    Args:
        X: DataFrame with time series data
        sample_indices: Array of unique sample indices
        max_lag: Maximum lag to check
        n_samples: Number of random samples to analyze
    """
    # Select random samples
    selected_samples = np.random.choice(
        sample_indices,
        min(n_samples, len(sample_indices)),
        replace=False
    )

    # Features to analyze (joint and survey features)
    joint_features = [col for col in X.columns if col.startswith('joint_')]
    survey_features = [col for col in X.columns if col.startswith('pain_survey_')]
    all_features = joint_features + survey_features

    print(f"Analyzing {len(selected_samples)} random samples...")
    print(f"Checking {len(all_features)} features")

    # Plot autocorrelation for each sample
    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    optimal_lags = []

    for idx, sample_idx in enumerate(selected_samples):
        # Get sample data
        sample_data = X[X['sample_index'] == sample_idx].sort_values('time')
        sample_data = sample_data[all_features]

        # Left plot: First feature
        if len(joint_features) > 0:
            feature_name = joint_features[0]
            plot_acf(
                sample_data[feature_name].values,
                lags=min(max_lag, len(sample_data)-1),
                ax=axes[idx, 0],
                alpha=0.05
            )
            axes[idx, 0].set_title(f'Sample {sample_idx} - {feature_name}')
            axes[idx, 0].axhline(y=0.2, color='red', linestyle='--', label='Threshold 0.2')
            axes[idx, 0].axhline(y=-0.2, color='red', linestyle='--')
            axes[idx, 0].legend()

        # Right plot: Average across all features
        avg_series = sample_data.mean(axis=1)
        plot_acf(
            avg_series.values,
            lags=min(max_lag, len(sample_data)-1),
            ax=axes[idx, 1],
            alpha=0.05
        )
        axes[idx, 1].set_title(f'Sample {sample_idx} - Average of all features')
        axes[idx, 1].axhline(y=0.2, color='red', linestyle='--', label='Threshold 0.2')
        axes[idx, 1].axhline(y=-0.2, color='red', linestyle='--')
        axes[idx, 1].legend()

        # Find where autocorrelation drops below 0.2
        acf_values = acf(avg_series.values, nlags=min(max_lag, len(sample_data)-1))
        significant_lags = np.where(np.abs(acf_values) > 0.2)[0]
        if len(significant_lags) > 1:
            optimal_lag = significant_lags[-1]
            optimal_lags.append(optimal_lag)
            print(f"  Sample {sample_idx}: Last significant lag at {optimal_lag}")

    plt.tight_layout()
    plt.show()

    # Recommend window size
    if optimal_lags:
        recommended_window = int(np.median(optimal_lags))
        print(f"\n{'='*60}")
        print(f" AUTOCORRELATION ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Significant lags found: {optimal_lags}")
        print(f"Median: {recommended_window}")
        print(f"Mean: {int(np.mean(optimal_lags))}")
        print(f"Range: {min(optimal_lags)} - {max(optimal_lags)}")
        print(f"\n RECOMMENDED WINDOW_SIZE: {recommended_window}")
        print(f"   (This is where autocorrelation drops below 0.2)")
        print(f"{'='*60}")
        return recommended_window
    else:
        print("\n Could not determine optimal window size from autocorrelation")
        return None


def apply_feature_engineering(df, roll=10):
    df = df.copy()

    joint_features = [c for c in df.columns if c.startswith("joint_")]

    for col in joint_features:
        df[col + "_diff"] = df.groupby("sample_index")[col].diff()
        df[col + "_vel"]  = df[col + "_diff"]
        df[col + "_acc"]  = df.groupby("sample_index")[col].diff().diff()

    for col in joint_features:
        df[col + f"_roll_mean_{roll}"] = (
            df.groupby("sample_index")[col]
              .rolling(roll)
              .mean()
              .reset_index(0, drop=True)
        )
        df[col + f"_roll_std_{roll}"] = (
            df.groupby("sample_index")[col]
              .rolling(roll)
              .std()
              .reset_index(0, drop=True)
        )

    df = df.bfill().ffill()

    return df

def compute_alpha_from_counts(counts):
    inv = 1 / np.array(counts, dtype=np.float32)
    alpha = inv / inv.sum()
    return alpha
