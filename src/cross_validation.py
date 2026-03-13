import os
import pickle
import numpy as np
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import apply_feature_engineering
from src.preprocessing import (
    preprocess_and_transform_datasets,
    build_dataloaders,
    create_sequences,
)
from src.models import RecurrentClassifier
from src.train import fit



def run_stratified_kfold_cv(
    df,
    criterion,
    device,
    k,
    epochs,
    batch_size,
    hidden_layers,
    hidden_size,
    learning_rate,
    dropout_rate,
    window_size,
    stride,
    rnn_type,
    bidirectional,
    num_workers=2,
    l1_lambda=0.0,
    weight_decay=0.0,
    patience=0,
    evaluation_metric="val_f1",
    mode="max",
    restore_best_weights=True,
    writer=None,
    verbose=10,
    seed=42,
    experiment_name="cross_validation",
):
    """
    Run stratified K-fold cross-validation at the sample level for the recurrent classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features, `sample_index`, and `label`.
    criterion : nn.Module
        Loss function used during training.
    device : torch.device
        Computation device.
    k : int
        Number of folds.
    epochs : int
        Number of training epochs per fold.
    batch_size : int
        Batch size.
    hidden_layers : int
        Number of recurrent layers.
    hidden_size : int
        Hidden size of the recurrent block.
    learning_rate : float
        Learning rate for AdamW.
    dropout_rate : float
        Dropout rate in the model.
    window_size : int
        Sequence window size.
    stride : int
        Sequence stride.
    rnn_type : str
        Type of recurrent block: "RNN", "LSTM", or "GRU".
    bidirectional : bool
        Whether to use a bidirectional recurrent block.
    num_workers : int, default=2
        Number of DataLoader workers.
    l1_lambda : float, default=0.0
        L1 regularization coefficient.
    weight_decay : float, default=0.0
        Weight decay for AdamW.
    patience : int, default=0
        Early stopping patience.
    evaluation_metric : str, default="val_f1"
        Metric tracked for early stopping.
    mode : str, default="max"
        Optimization direction for early stopping.
    restore_best_weights : bool, default=True
        Whether to restore best fold weights.
    writer : SummaryWriter or None
        Optional TensorBoard writer.
    verbose : int, default=10
        Print frequency.
    seed : int, default=42
        Random seed.
    experiment_name : str, default="cross_validation"
        Prefix used for saved fold artifacts.

    Returns
    -------
    fold_losses : dict
    fold_metrics : dict
    best_scores : dict
    """

    fold_losses = {}
    fold_metrics = {}
    best_scores = {}

    experiment_dir = os.path.join("models", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    df_samples = df[["sample_index", "label"]].drop_duplicates()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_samples["sample_index"], df_samples["label"])
    ):
        train_samples = df_samples.iloc[train_idx]["sample_index"].values
        val_samples = df_samples.iloc[val_idx]["sample_index"].values

        with open(os.path.join(experiment_dir, f"fold_{fold}_val_samples.pkl"), "wb") as f:
            pickle.dump(val_samples, f)

        with open(os.path.join(experiment_dir, f"fold_{fold}_train_samples.pkl"), "wb") as f:
            pickle.dump(train_samples, f)

        if verbose > 0:
            print("\n" + "=" * 60)
            print(
                f"Fold {fold + 1}/{k} | "
                f"train samples: {len(train_samples)} | val samples: {len(val_samples)}"
            )
            print(
                f"window={window_size}, stride={stride}, hidden_size={hidden_size}, "
                f"bidirectional={bidirectional}, rnn_type={rnn_type}"
            )
            print("=" * 60)

        df_train_raw = df[df["sample_index"].isin(train_samples)].copy()
        df_val_raw = df[df["sample_index"].isin(val_samples)].copy()

        X_train_features_raw = df_train_raw.drop(columns=["label"])
        y_train_labels = df_train_raw[["sample_index", "label"]].drop_duplicates(subset=["sample_index"])

        X_val_features_raw = df_val_raw.drop(columns=["label"])
        y_val_labels = df_val_raw[["sample_index", "label"]].drop_duplicates(subset=["sample_index"])

        X_train_features_raw = apply_feature_engineering(X_train_features_raw)
        X_val_features_raw = apply_feature_engineering(X_val_features_raw)

        X_train_processed, X_val_processed, _, preprocessing_pipeline = (
            preprocess_and_transform_datasets(
                X_train_features_raw,
                X_val_features_raw,
                None 
            )
        )

        with open(os.path.join(experiment_dir, f"fold_{fold}_preprocess.pkl"), "wb") as f:
            pickle.dump(preprocessing_pipeline, f)

        label_encoder = LabelEncoder()
        label_encoder.fit(y_train_labels["label"])

        with open(os.path.join(experiment_dir, f"fold_{fold}_label_encoder.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)

        train_sample_indices = X_train_processed["sample_index"].unique()
        train_sequences, train_labels, _ = create_sequences(
            X_train_processed,
            y_train_labels,
            train_sample_indices,
            window_size,
            stride,
            label_encoder,
        )

        val_sample_indices = X_val_processed["sample_index"].unique()
        val_sequences, val_labels, _ = create_sequences(
            X_val_processed,
            y_val_labels,
            val_sample_indices,
            window_size,
            stride,
            label_encoder,
        )

        input_shape = train_sequences.shape[1:]
        np.save(os.path.join(experiment_dir, f"fold_{fold}_input_shape.npy"), np.array(input_shape))

        num_classes = len(np.unique(train_labels))
        with open(os.path.join(experiment_dir, f"fold_{fold}_num_classes.pkl"), "wb") as f:
            pickle.dump(num_classes, f)

        _, _, train_loader, val_loader = build_dataloaders(
            train_sequences,
            train_labels,
            val_sequences,
            val_labels,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        model = RecurrentClassifier(
            input_size=input_shape[-1],
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )

        split_scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        model, training_history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer,
            scaler=split_scaler,
            device=device,
            l1_lambda=l1_lambda,
            writer=None,
            patience=patience,
            verbose=verbose,
            evaluation_metric=evaluation_metric,
            mode=mode,
            restore_best_weights=restore_best_weights,
            experiment_name=f"{experiment_name}/split_{fold}",
            scheduler=scheduler,
        )

        fold_losses[f"split_{fold}"] = training_history["val_loss"]
        fold_metrics[f"split_{fold}"] = training_history["val_f1"]
        best_scores[f"split_{fold}"] = max(training_history["val_f1"])

    best_scores["mean"] = np.mean(
        [best_scores[key] for key in best_scores if key.startswith("split_")]
    )
    best_scores["std"] = np.std(
        [best_scores[key] for key in best_scores if key.startswith("split_")]
    )

    if verbose > 0:
        print(f"Best score: {best_scores['mean']:.4f} ± {best_scores['std']:.4f}")

    return fold_losses, fold_metrics, best_scores
