import os
import pandas as pd

main_dir = "MobiAct_Dataset_v2.0\MobiAct_Dataset_v2.0\Annotated Data"  # folder containing WAL, STD, etc.

for folder in os.listdir(main_dir):
    folder_path = os.path.join(main_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)
                
                # count missing values per column
                missing_counts = df.isna().sum()
                total_missing = missing_counts.sum()
                
                if total_missing > 0:
                    print(f"File: {file_path}")
                    print("Missing values per column:")
                    print(missing_counts[missing_counts > 0])
                    print("---")
