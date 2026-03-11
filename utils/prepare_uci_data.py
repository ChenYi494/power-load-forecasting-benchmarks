import os
import pandas as pd
from config.params import BASE_DIR, PATH

def prepare_uci_data():
    # Load raw dataset
    # Handling European numeric formatting (decimal=',') and semicolon delimiters
    file_path = 'LD2011_2014.txt'
    df = pd.read_csv(file_path, sep=';', decimal=',', low_memory=False)

    # Standardize column naming and timestamp parsing
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Target User Selection: Filtering specific load profiles (e.g., MT_200)
    # Filter out zero values (pre-installation data points) to ensure data integrity
    target_user = 'MT_200'
    df_clean = df[['timestamp', target_user]].copy()
    df_clean.rename(columns={target_user: 'load'}, inplace=True)
    df_clean = df_clean[df_clean['load'] > 0].reset_index(drop=True)

    # Temporal Feature Engineering (15-minute resolution)
    # Extracting cyclical time features to capture daily and weekly seasonality
    df_clean['hour'] = df_clean['timestamp'].dt.hour
    df_clean['minute'] = df_clean['timestamp'].dt.minute
    df_clean['day_of_week'] = df_clean['timestamp'].dt.dayofweek
    df_clean['is_weekend'] = (df_clean['day_of_week'] >= 5).astype(int)

    # Autoregressive Lag Features
    # lag_1: Immediate prior step (15 mins ago)
    # lag_96: Seasonal prior step (24 hours ago, 96 intervals of 15 mins)
    df_clean['lag_1'] = df_clean['load'].shift(1)
    df_clean['lag_96'] = df_clean['load'].shift(96)

    # Drop NaN values resulting from the shifting operations
    df_clean.dropna(inplace=True)

    # Slice the dataset for experimental consistency
    df_final = df_clean.head(110000)

    # Persistence: Exporting to Parquet format with Snappy compression
    # Parquet is used for efficient I/O and schema preservation in ML pipelines
    output_path = os.path.join(BASE_DIR, PATH['dataset_folder'], PATH['source_data'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_parquet(output_path, compression='snappy', index=False)

    print(f"Preprocessing complete. Industrial-standard Parquet generated: {output_path}")
    print(f"Record Count: {len(df_final)}")
    print(f"Temporal Span: {df_final['timestamp'].min()} to {df_final['timestamp'].max()}")

prepare_uci_data()