import glob
import os
import pandas as pd
from pathlib import Path


# Data preprocessing function
def data_preprocess():
    csv_files = []
    folder = Path("../Data")  # Directory
    for file in folder.glob("*.csv"):
        csv_files.append(os.path.join(folder, file.name))
    print(csv_files)
    df_list = []
    for file in csv_files:
        # Read CSV file
        df = pd.read_csv(file)
        # Optional: Add a column to record the source file name for tracking data source later
        df_list.append(df)

    for i in range(len(df_list)):
        df_list[i].columns = ['Sentence', 'Importance','Longstorage']
        
    # Merge DataFrames, ignore_index=True will reindex
    combined_df = pd.concat(df_list, ignore_index=True, axis=0)

    # Use sample method to shuffle rows, frac=1 means take all data
    sentence_df = combined_df.sample(frac=1).reset_index(drop=True)
    

    sentence_df.columns = ['Sentence', 'Importance','Longstorage']

    replace_dict = {"是": 1, "yes": 1, "Yes": 1, "否": 0, "no": 0, "No": 0}

    sentence_df.iloc[:, 2] = sentence_df.iloc[:, 2].replace(replace_dict)

    return sentence_df
