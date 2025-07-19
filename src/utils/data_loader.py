import os
import pandas as pd

current_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT,"local", "data","raw","physionet.org","files","q-pain","1.0.0")

def load_raw_data():
    csv_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')]

    all_dataframes = []

    for csv_file in csv_files:
        file_path = os.path.join(RAW_DATA_PATH, csv_file)
        df = pd.read_csv(file_path)
        df['source_file'] = csv_file  # Add a column with the file name
        all_dataframes.append(df)
        print(f"Loaded {csv_file} with shape {df.shape}")

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")

    return combined_df