import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

from src.feature_engineering import apply_feature_engineering
from src.preprocessing import create_sequences
from src.models import RecurrentClassifier

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

def plot_confusion_for_fold(
    df,
    best_config,
    fold,
    hidden_layers,
    rnn_type,
    model_root="models",
    batch_size=256,
    num_workers=0,
    device=None,
):
    """
    Plot the confusion matrix for the validation set of a given fold.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe containing features, `sample_index`, and `label`.
    best_config : dict
        Best hyperparameter configuration found during tuning.
    fold : int
        Fold index to evaluate.
    hidden_layers : int
        Number of recurrent layers used by the model.
    rnn_type : str
        Recurrent architecture type ("RNN", "LSTM", or "GRU").
    model_root : str, default="models"
        Root directory where fold artifacts and trained models are stored.
    batch_size : int, default=256
        Batch size for validation inference.
    num_workers : int, default=0
        Number of DataLoader workers.
    device : torch.device or None
        Computation device. If None, automatically detected.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix for the selected fold.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n--- Confusion matrix for fold {fold} ---")

    config_str = "_".join([f"{k}_{v}" for k, v in best_config.items()])
    experiment_dir = os.path.join(model_root, config_str)
    best_model_path = os.path.join(experiment_dir, f"split_{fold}_model.pt")

    print(f"Using config_str: {config_str}")
    print(f"Model path: {best_model_path}")

    val_samples = pickle.load(open(os.path.join(experiment_dir, f"fold_{fold}_val_samples.pkl"), "rb"))
    preprocessing_pipeline = pickle.load(open(os.path.join(experiment_dir, f"fold_{fold}_preprocess.pkl"), "rb"))
    input_shape = np.load(os.path.join(experiment_dir, f"fold_{fold}_input_shape.npy"))
    num_classes = pickle.load(open(os.path.join(experiment_dir, f"fold_{fold}_num_classes.pkl"), "rb"))
    label_encoder = pickle.load(open(os.path.join(experiment_dir, f"fold_{fold}_label_encoder.pkl"), "rb"))

    class_names = label_encoder.classes_

    df_val_raw = df[df["sample_index"].isin(val_samples)].copy()
    X_val_raw = df_val_raw.drop(columns=["label"])

    X_val_raw = apply_feature_engineering(X_val_raw)
    X_val_processed = preprocessing_pipeline.transform(X_val_raw)

    fitted_preprocessor = preprocessing_pipeline.named_steps["preprocessor"]
    numerical_features = fitted_preprocessor.transformers_[0][2]
    categorical_features = fitted_preprocessor.transformers_[1][2]

    categorical_feature_names = (
        fitted_preprocessor.named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )

    processed_feature_names = numerical_features + categorical_feature_names.tolist()

    X_val_processed = pd.DataFrame(
        X_val_processed,
        columns=processed_feature_names,
        index=X_val_raw.index,
    )

    X_val_processed["sample_index"] = X_val_raw["sample_index"]
    X_val_processed["time"] = X_val_raw["time"]

    y_val_labels = (
        df_val_raw[["sample_index", "label"]]
        .drop_duplicates(subset=["sample_index"])
        .copy()
    )

    val_sample_indices = X_val_processed["sample_index"].unique()

    val_sequences, val_labels, _ = create_sequences(
        X_val_processed,
        y_val_labels,
        val_sample_indices,
        best_config["window_size"],
        best_config["stride"],
        label_encoder,
    )

    val_ds = TensorDataset(
        torch.from_numpy(val_sequences).float(),
        torch.from_numpy(val_labels).long(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = RecurrentClassifier(
        input_size=input_shape[-1],
        hidden_size=best_config["hidden_size"],
        num_layers=hidden_layers,
        num_classes=num_classes,
        dropout_rate=best_config["dropout_rate"],
        bidirectional=best_config["bidirectional"],
        rnn_type=rnn_type,
    ).to(device)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(inputs)

            preds = outputs.argmax(dim=1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_val_labels, all_val_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix - Fold {fold} - {config_str}")
    plt.show()

    return cm

def plot_aggregated_confusion_matrix_cv(
    df,
    best_config,
    k,
    hidden_layers,
    rnn_type,
    model_root="models",
    batch_size=256,
    num_workers=0,
    device=None,
):
    """
    Aggregate out-of-fold predictions across all validation folds and
    plot a single confusion matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe containing features, `sample_index`, and `label`.
    best_config : dict
        Best hyperparameter configuration found during tuning.
    k : int
        Number of cross-validation folds.
    hidden_layers : int
        Number of recurrent layers.
    rnn_type : str
        Type of recurrent architecture ("RNN", "LSTM", or "GRU").
    model_root : str, default="models"
        Root directory where model artifacts are stored.
    batch_size : int, default=256
        Batch size for inference.
    num_workers : int, default=0
        Number of DataLoader workers.
    device : torch.device or None
        Computation device.

    Returns
    -------
    cm : np.ndarray
        Aggregated confusion matrix across all folds.
    all_true : np.ndarray
        True labels aggregated across folds.
    all_pred : np.ndarray
        Predicted labels aggregated across folds.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_str = "_".join([f"{k}_{v}" for k, v in best_config.items()])
    experiment_dir = os.path.join(model_root, config_str)

    all_true = []
    all_pred = []
    class_names = None

    for fold in range(k):
        print(f"\nProcessing fold {fold + 1}/{k}...")

        val_samples = pickle.load(
            open(os.path.join(experiment_dir, f"fold_{fold}_val_samples.pkl"), "rb")
        )
        preprocessing_pipeline = pickle.load(
            open(os.path.join(experiment_dir, f"fold_{fold}_preprocess.pkl"), "rb")
        )
        input_shape = np.load(
            os.path.join(experiment_dir, f"fold_{fold}_input_shape.npy")
        )
        num_classes = pickle.load(
            open(os.path.join(experiment_dir, f"fold_{fold}_num_classes.pkl"), "rb")
        )
        label_encoder = pickle.load(
            open(os.path.join(experiment_dir, f"fold_{fold}_label_encoder.pkl"), "rb")
        )

        class_names = label_encoder.classes_

        best_model_path = os.path.join(experiment_dir, f"split_{fold}_model.pt")

        df_val_raw = df[df["sample_index"].isin(val_samples)].copy()
        X_val_raw = df_val_raw.drop(columns=["label"])

        X_val_raw = apply_feature_engineering(X_val_raw)
        X_val_processed = preprocessing_pipeline.transform(X_val_raw)

        fitted_preprocessor = preprocessing_pipeline.named_steps["preprocessor"]
        numerical_features = fitted_preprocessor.transformers_[0][2]
        categorical_features = fitted_preprocessor.transformers_[1][2]

        categorical_feature_names = (
            fitted_preprocessor.named_transformers_["cat"]
            .get_feature_names_out(categorical_features)
        )

        processed_feature_names = numerical_features + categorical_feature_names.tolist()

        X_val_processed = pd.DataFrame(
            X_val_processed,
            columns=processed_feature_names,
            index=X_val_raw.index,
        )

        X_val_processed["sample_index"] = X_val_raw["sample_index"]
        X_val_processed["time"] = X_val_raw["time"]

        y_val_labels = (
            df_val_raw[["sample_index", "label"]]
            .drop_duplicates(subset=["sample_index"])
            .copy()
        )

        val_sample_indices = X_val_processed["sample_index"].unique()

        val_sequences, val_labels, _ = create_sequences(
            X_val_processed,
            y_val_labels,
            val_sample_indices,
            best_config["window_size"],
            best_config["stride"],
            label_encoder,
        )

        val_ds = TensorDataset(
            torch.from_numpy(val_sequences).float(),
            torch.from_numpy(val_labels).long(),
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = RecurrentClassifier(
            input_size=input_shape[-1],
            hidden_size=best_config["hidden_size"],
            num_layers=hidden_layers,
            num_classes=num_classes,
            dropout_rate=best_config["dropout_rate"],
            bidirectional=best_config["bidirectional"],
            rnn_type=rnn_type,
        ).to(device)

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        fold_true = []
        fold_pred = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                ):
                    outputs = model(inputs)

                preds = outputs.argmax(dim=1)
                fold_pred.extend(preds.cpu().numpy())
                fold_true.extend(labels.cpu().numpy())

        all_true.extend(fold_true)
        all_pred.extend(fold_pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    cm = confusion_matrix(all_true, all_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Aggregated Confusion Matrix Across Validation Folds")
    plt.show()

    print("\nClassification report (aggregated across folds):")
    print(classification_report(all_true, all_pred, target_names=class_names))

    return cm, all_true, all_pred

def build_submission_from_sequence_predictions(sample_indices, predictions, label_encoder):
    """
    Aggregate sequence-level predictions into sample-level predictions
    using majority vote.
    """
    df_submission = pd.DataFrame(
        {
            "sample_index": sample_indices,
            "label": [label_encoder.inverse_transform([p])[0] for p in predictions],
        }
    )

    final_submission = (
        df_submission.groupby("sample_index")["label"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )

    return final_submission

def plot_aggregated_roc_auc_cv(
    df,
    best_config,
    k,
    hidden_layers,
    rnn_type,
    model_root="models",
    batch_size=256,
    num_workers=0,
    device=None,
):
    """
    Aggregate out-of-fold predicted probabilities across validation folds
    and plot multiclass ROC curves (one-vs-rest).

    Returns
    -------
    roc_auc_dict : dict
        AUC per class plus macro/micro averages.
    all_true : np.ndarray
        Aggregated true labels across folds.
    all_probs : np.ndarray
        Aggregated predicted probabilities across folds.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_str = "_".join([f"{k}_{v}" for k, v in best_config.items()])
    experiment_dir = os.path.join(model_root, config_str)

    all_true = []
    all_probs = []
    class_names = None

    for fold in range(k):
        print(f"\nProcessing fold {fold + 1}/{k} for ROC/AUC...")

        val_samples = pickle.load(
            open(os.path.join(experiment_dir, f"fold_{fold}_val_samples.pkl"), "rb")
        )
        preprocessing_pipeline = pickle.load(
            open(os.path.join(experiment_dir, f"fold_{fold}_preprocess.pkl"), "rb")
        )
        input_shape = np.load(
            os.path.join(experiment_dir, f"fold_{fold}_input_shape.npy")
        )
        num_classes = pickle.load(
            open(os.path.join(experiment_dir, f"fold_{fold}_num_classes.pkl"), "rb")
        )
        label_encoder = pickle.load(
            open(os.path.join(experiment_dir, f"fold_{fold}_label_encoder.pkl"), "rb")
        )

        class_names = label_encoder.classes_
        best_model_path = os.path.join(experiment_dir, f"split_{fold}_model.pt")

        df_val_raw = df[df["sample_index"].isin(val_samples)].copy()
        X_val_raw = df_val_raw.drop(columns=["label"])

        X_val_raw = apply_feature_engineering(X_val_raw)
        X_val_processed = preprocessing_pipeline.transform(X_val_raw)

        fitted_preprocessor = preprocessing_pipeline.named_steps["preprocessor"]
        numerical_features = fitted_preprocessor.transformers_[0][2]
        categorical_features = fitted_preprocessor.transformers_[1][2]

        categorical_feature_names = (
            fitted_preprocessor.named_transformers_["cat"]
            .get_feature_names_out(categorical_features)
        )

        processed_feature_names = numerical_features + categorical_feature_names.tolist()

        X_val_processed = pd.DataFrame(
            X_val_processed,
            columns=processed_feature_names,
            index=X_val_raw.index,
        )

        X_val_processed["sample_index"] = X_val_raw["sample_index"]
        X_val_processed["time"] = X_val_raw["time"]

        y_val_labels = (
            df_val_raw[["sample_index", "label"]]
            .drop_duplicates(subset=["sample_index"])
            .copy()
        )

        val_sample_indices = X_val_processed["sample_index"].unique()

        val_sequences, val_labels, _ = create_sequences(
            X_val_processed,
            y_val_labels,
            val_sample_indices,
            best_config["window_size"],
            best_config["stride"],
            label_encoder,
        )

        val_ds = TensorDataset(
            torch.from_numpy(val_sequences).float(),
            torch.from_numpy(val_labels).long(),
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = RecurrentClassifier(
            input_size=input_shape[-1],
            hidden_size=best_config["hidden_size"],
            num_layers=hidden_layers,
            num_classes=num_classes,
            dropout_rate=best_config["dropout_rate"],
            bidirectional=best_config["bidirectional"],
            rnn_type=rnn_type,
        ).to(device)

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        fold_true = []
        fold_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                ):
                    logits = model(inputs)

                probs = torch.softmax(logits, dim=1)

                fold_true.extend(labels.cpu().numpy())
                fold_probs.extend(probs.cpu().numpy())

        all_true.extend(fold_true)
        all_probs.extend(fold_probs)

    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    y_bin = label_binarize(all_true, classes=np.arange(all_probs.shape[1]))

    roc_auc_dict = {}
    fpr = {}
    tpr = {}

    for i in range(all_probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc_dict[class_names[i]] = auc(fpr[i], tpr[i])

    roc_auc_dict["macro"] = roc_auc_score(
        y_bin, all_probs, average="macro", multi_class="ovr"
    )
    roc_auc_dict["micro"] = roc_auc_score(
        y_bin, all_probs, average="micro", multi_class="ovr"
    )

    plt.figure(figsize=(8, 6))
    for i in range(all_probs.shape[1]):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"{class_names[i]} (AUC = {roc_auc_dict[class_names[i]]:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Aggregated ROC Curves Across Validation Folds")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    print("\nAggregated ROC-AUC scores:")
    for key, value in roc_auc_dict.items():
        print(f"{key}: {value:.4f}")

    return roc_auc_dict, all_true, all_probs
