import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset from Parquet, perform temporal splitting, and standardize features.
def load_dataset(dataset_path, target_col, test_ratio=0.2, dataset_size=0, shuffle=False):
    """
    Args:
        dataset_path: Path to the source Parquet file
        target_col: The dependent variable (label) for forecasting
        test_ratio: Proportion of data reserved for the final test set
        dataset_size: Number of samples to include (used for scalability profiling)
        shuffle: Whether to shuffle data (False to prevent temporal leakage)
    """

    # Load data from memory-efficient Parquet format
    df = pd.read_parquet(dataset_path)

    # Scale dataset for performance profiling across different magnitudes
    if dataset_size > 0:
        df = df.iloc[:dataset_size]

    # Clean column headers to prevent key errors
    df.columns = df.columns.str.strip()

    # Feature engineering: Isolate numerical features (X) and target variable (y)
    X = df.select_dtypes(include=['number']).drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    # Sequential Data Splitting
    # First Split: Reserve the latest temporal segment as the Hold-out Test Set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=shuffle
    )

    # Second Split: Carve out a Validation Set from the training pool (10%)
    # Validation data is used for hyperparameter tuning and Early Stopping triggers
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, shuffle=shuffle
    )

    # Feature Scaling (Z-score Normalization)
    # Prevents models from being biased toward features with larger absolute magnitudes
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Return standardized datasets and the original dataframe for reference
    return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values, df


    # =====================================================================================================
    # DATA PARTITIONING STRATEGY:
    # 1. Training Set   : Used for parameter optimization and weight updates.
    # 2. Validation Set : Used for real-time performance monitoring and Early Stopping in MLP.
    # 3. Test Set       : The 'final exam' for evaluating model generalization on unseen future data.
    # =====================================================================================================