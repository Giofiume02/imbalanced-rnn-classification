import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from src.feature_engineering import compute_alpha_from_counts

def create_sequences(X, y, sample_indices, window_size, stride, label_encoder=None):
    """
    Creates sequences from the processed data for each sample index.

    Args:
        X (pd.DataFrame): The processed feature DataFrame.
        y (pd.DataFrame): The labels DataFrame. Can be None or empty for test set.
        sample_indices (np.ndarray): Array of unique sample indices.
        window_size (int): The size of the time window for each sequence.
        stride (int): The step size to move the window.
        label_encoder (LabelEncoder or None): Fitted LabelEncoder for encoding labels.

    Returns:
        tuple: A tuple containing:
            - sequences (np.ndarray): Array of sequences.
            - sequence_labels (np.ndarray or None): Array of encoded integer labels for each sequence, or None if y is None or empty.
            - sequence_sample_indices (list): List of sample indices for each sequence.
    """
    sequences = []
    sequence_labels = []
    sequence_sample_indices = [] 

    y_provided = y is not None and not y.empty
    if y_provided:
        y_indexed = y.set_index('sample_index')


    for sample_index in sample_indices:
        
        sample_data = X[X['sample_index'] == sample_index].drop(columns=['sample_index', 'time']) 

        sample_label = None
        if y_provided:
            try:
                
                temp_label = y_indexed.loc[sample_index, 'label']
                
                if isinstance(temp_label, pd.Series):
                    sample_label = temp_label.iloc[0]
                else:
                    sample_label = temp_label
                    
            except KeyError:
                
                print(f"Warning: sample_index {sample_index} not found in y_indexed.")
                continue 

        if len(sample_data) >= window_size:
            for i in range(0, len(sample_data) - window_size + 1, stride):
                sequence = sample_data.iloc[i : i + window_size]
                sequences.append(sequence.values)

               
                if y_provided:
                    sequence_labels.append(sample_label)

                sequence_sample_indices.append(sample_index) 


    if sequences: 
        sequences = np.array(sequences)
    else:
        sequences = np.array([]) 

    if y_provided and sequence_labels and label_encoder is not None:
        
        sequence_labels_encoded = label_encoder.transform(sequence_labels)
        sequence_labels = np.array(sequence_labels_encoded)
    else:
        sequence_labels = None 


    return sequences, sequence_labels, sequence_sample_indices 


def convert_joint_columns_to_float32(df):
    """
    Convert joint angle columns to float32 to reduce memory usage
    and ensure consistent numeric types.
    """
    df = df.copy()

    for i in range(31):
        col = f"joint_{i:02d}"
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    return df


def remove_zero_variance_columns(train_df, test_df):
    """
    Remove columns with zero variance based on the training set,
    and apply the same removal to the test set.
    """
    zero_var_cols = train_df.loc[:, train_df.nunique() == 1].columns.tolist()

    train_df = train_df.drop(columns=zero_var_cols)
    test_df = test_df.drop(columns=zero_var_cols)

    return train_df, test_df, zero_var_cols


def remove_high_correlation_features(train_df, test_df, threshold=0.90):
    """
    Remove highly correlated numerical features based on the training set,
    and apply the same removal to the test set.
    """
    num_cols = train_df.select_dtypes(include=np.number).columns
    corr = train_df[num_cols].corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] > threshold)]

    train_df = train_df.drop(columns=high_corr_cols)
    test_df = test_df.drop(columns=high_corr_cols)

    return train_df, test_df, high_corr_cols

def fit_label_encoder_and_compute_alpha(y_train, device):
    """
    Fit a LabelEncoder on unique sample-level labels and compute
    inverse-frequency class weights for focal loss.

    Returns
    -------
    label_encoder : LabelEncoder
    counts : np.ndarray
    alpha_np : np.ndarray
    alpha_tensor : torch.Tensor
    """
    df_seq = y_train[["sample_index", "label"]].drop_duplicates(subset=["sample_index"])

    label_encoder = LabelEncoder()
    label_encoder.fit(df_seq["label"])

    counts = np.array(
        [(df_seq["label"] == c).sum() for c in label_encoder.classes_],
        dtype=np.float32,
    )

    alpha_np = compute_alpha_from_counts(counts)
    alpha_tensor = torch.tensor(alpha_np, dtype=torch.float32).to(device)

    return label_encoder, counts, alpha_np, alpha_tensor


def split_train_validation_by_sample(X_train, y_train, test_size=0.3, random_state=42):
    """
    Split train and validation sets at the sample level to avoid leakage.

    Returns
    -------
    X_train_split, y_train_split, X_val_split, y_val_split
    """
    df_samples = y_train[["sample_index", "label"]].drop_duplicates()

    train_samples, val_samples = train_test_split(
        df_samples,
        test_size=test_size,
        random_state=random_state,
        stratify=df_samples["label"],
    )

    train_sample_indices = train_samples["sample_index"].values
    val_sample_indices = val_samples["sample_index"].values

    X_train_split = X_train[X_train["sample_index"].isin(train_sample_indices)].copy()
    y_train_split = y_train[y_train["sample_index"].isin(train_sample_indices)].copy()

    X_val_split = X_train[X_train["sample_index"].isin(val_sample_indices)].copy()
    y_val_split = y_train[y_train["sample_index"].isin(val_sample_indices)].copy()

    return X_train_split, y_train_split, X_val_split, y_val_split


def preprocess_and_transform_datasets(X_train_split, X_val_split, X_test=None):
    """
    Fit preprocessing transformations on the training split and apply them
    consistently to training, validation, and optionally test sets.

    Parameters
    ----------
    X_train_split : pd.DataFrame
        Training features.
    X_val_split : pd.DataFrame
        Validation features.
    X_test : pd.DataFrame or None, default=None
        Optional test features.

    Returns
    -------
    X_train_processed : pd.DataFrame
    X_val_processed : pd.DataFrame
    X_test_processed : pd.DataFrame or None
    preprocessing_pipeline : sklearn.pipeline.Pipeline
    """
    numerical_features = X_train_split.select_dtypes(include=np.number).columns.tolist()
    numerical_features = [col for col in numerical_features if col not in ["sample_index", "time"]]

    categorical_features = X_train_split.select_dtypes(include="object").columns.tolist()

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    preprocessing_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    preprocessing_pipeline.fit(X_train_split)

    X_train_processed = preprocessing_pipeline.transform(X_train_split)
    X_val_processed = preprocessing_pipeline.transform(X_val_split)
    X_test_processed = preprocessing_pipeline.transform(X_test) if X_test is not None else None

    fitted_preprocessor = preprocessing_pipeline.named_steps["preprocessor"]
    numerical_feature_names = numerical_features
    categorical_feature_names = (
        fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    )

    processed_feature_names = numerical_feature_names + categorical_feature_names.tolist()

    X_train_processed = pd.DataFrame(
        X_train_processed, columns=processed_feature_names, index=X_train_split.index
    )
    X_val_processed = pd.DataFrame(
        X_val_processed, columns=processed_feature_names, index=X_val_split.index
    )

    if X_test is not None:
        X_test_processed = pd.DataFrame(
            X_test_processed, columns=processed_feature_names, index=X_test.index
        )

    X_train_processed["sample_index"] = X_train_split["sample_index"]
    X_train_processed["time"] = X_train_split["time"]

    X_val_processed["sample_index"] = X_val_split["sample_index"]
    X_val_processed["time"] = X_val_split["time"]

    if X_test is not None:
        X_test_processed["sample_index"] = X_test["sample_index"]
        X_test_processed["time"] = X_test["time"]

    return X_train_processed, X_val_processed, X_test_processed, preprocessing_pipeline


def build_dataloaders(train_sequences, train_labels, val_sequences, val_labels, batch_size=64, num_workers=2):
    """
    Build TensorDatasets and DataLoaders for training and validation.
    """
    train_ds = TensorDataset(
        torch.from_numpy(train_sequences).float(),
        torch.from_numpy(train_labels).long(),
    )

    val_ds = TensorDataset(
        torch.from_numpy(val_sequences).float(),
        torch.from_numpy(val_labels).long(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_ds, val_ds, train_loader, val_loader

def preprocess_full_train_and_test(X_train_full, X_test):
    """
    Fit preprocessing transformations on the full training set and apply them
    to both training and test data.

    Returns
    -------
    X_train_processed : pd.DataFrame
    X_test_processed : pd.DataFrame
    preprocessing_pipeline : sklearn.pipeline.Pipeline
    """
    numerical_features = X_train_full.select_dtypes(include=np.number).columns.tolist()
    numerical_features = [col for col in numerical_features if col not in ["sample_index", "time"]]

    categorical_features = X_train_full.select_dtypes(include="object").columns.tolist()

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    preprocessing_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    preprocessing_pipeline.fit(X_train_full)

    X_train_processed = preprocessing_pipeline.transform(X_train_full)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    fitted_preprocessor = preprocessing_pipeline.named_steps["preprocessor"]
    numerical_feature_names = numerical_features
    categorical_feature_names = (
        fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    )

    processed_feature_names = numerical_feature_names + categorical_feature_names.tolist()

    X_train_processed = pd.DataFrame(
        X_train_processed, columns=processed_feature_names, index=X_train_full.index
    )
    X_test_processed = pd.DataFrame(
        X_test_processed, columns=processed_feature_names, index=X_test.index
    )

    X_train_processed["sample_index"] = X_train_full["sample_index"]
    X_train_processed["time"] = X_train_full["time"]

    X_test_processed["sample_index"] = X_test["sample_index"]
    X_test_processed["time"] = X_test["time"]

    return X_train_processed, X_test_processed, preprocessing_pipeline
