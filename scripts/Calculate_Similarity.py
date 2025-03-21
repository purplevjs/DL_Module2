import os
print(os.listdir('/kaggle/input/text-similarity'))
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
import numpy as np
import pandas as pd


# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df_train = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "violetsuh/text-similarity",
  "archive (7)/train.csv"
)

df_test = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "violetsuh/text-similarity",
  "archive (7)/test.csv"
)

print("Train's First 5 records:", df_train.head())
print("Test's First 5 records:", df_test.head())

df_train["dataset"] = "train"
df_test["dataset"] = "test"

df_all = pd.concat([df_train, df_test], ignore_index=True)

print("Dataset Loaded:")
print(df_all.head())

text_columns = df_all.select_dtypes(include=["object"]).columns.tolist()
text_columns.remove("dataset")  #remove the label

print("Text columns to encode: ", text_columns)

for col in text_columns:
    print(f"Column: {col}")
    print(df_all[col].value_counts().head(10))  # Show top 10 most frequent values
    print("\n" + "="*50 + "\n")
    
for col in text_columns:
    df_all[col] = df_all[col].astype(str).fillna("")


for col in text_columns:
    print(f"Encoding column: {col}")
    df_all[f"{col}_embedding"] = list(model.encode(df_all[col].tolist(), convert_to_numpy=True))

df_all.head()

from sklearn.metrics.pairwise import cosine_similarity

embedding_columns = [col for col in df_all.columns if col.endswith("_embedding")]

embeddings_matrix = np.stack(df_all[embedding_columns].apply(lambda row: np.concatenate(row), axis=1))

similarity_matrix = cosine_similarity(embeddings_matrix)

np.fill_diagonal(similarity_matrix, 0)

most_similar_index = np.argmax(similarity_matrix.sum(axis=1))





df_cleaned = df_all.drop(index=most_similar_index).reset_index(drop=True)

print(f"Removed row at index: {most_similar_index}")


df_cleaned.head()