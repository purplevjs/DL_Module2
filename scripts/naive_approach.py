import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from Data_preprocess import data_preprocess

# Create a folder to save results
results_folder = "naive_approach_results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Define simple words for matching
simple_words = ['have', 'own', 'possess', 'hold', 'like', 'love', 'prefer', 'fancy', 'admire', 'enjoy', 'savor', 'relish', 'appreciate', 'delight in']

# Load data
sentence_df = data_preprocess()

# Check column names
print("DataFrame columns:", sentence_df.columns.tolist())

# Assuming the first column is text and third column is binary classification
text_column = sentence_df.columns[0]  # First column for text
label_column = sentence_df.columns[2]  # Third column for classification label

# Function to normalize yes/no to binary
def normalize_yes_no(value):
    if isinstance(value, (int, float)):
        return 1 if value == 1 else 0
    elif isinstance(value, str):
        value = value.strip().lower()
        if value in ['yes', 'y', 'si', 'sí', 'oui', 'ja', '是', '是的']:
            return 1
        elif value in ['no', 'n', 'non', 'nein', '否', '不']:
            return 0
        else:
            print(f"Warning: Unrecognized value '{value}', setting to -1")
            return -1
    else:
        return -1

# Convert labels to binary format
sentence_df['binary_label'] = sentence_df[label_column].apply(normalize_yes_no)

# Remove entries with invalid labels
sentence_df = sentence_df[sentence_df['binary_label'] != -1]

# Function to check if any simple word is in the text
def contains_simple_word(text):
    if not isinstance(text, str):
        return 0
    
    text_lower = text.lower()
    for word in simple_words:
        if word in text_lower:
            return 1
    return 0

# Create predictions based on simple word matching
sentence_df['prediction'] = sentence_df[text_column].apply(contains_simple_word)

# Get true values and predictions
true_values = sentence_df['binary_label']
predictions = sentence_df['prediction']

# Calculate metrics
accuracy = accuracy_score(true_values, predictions)
precision = precision_score(true_values, predictions, zero_division=0)
recall = recall_score(true_values, predictions, zero_division=0)
f1 = f1_score(true_values, predictions, zero_division=0)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Create confusion matrix
cm = confusion_matrix(true_values, predictions)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No (0)', 'Yes (1)'],
            yticklabels=['No (0)', 'Yes (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Simple Word Matching Approach')
plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'))
plt.close()

# Save predictions and true values to CSV
results_df = pd.DataFrame({
    'Text': sentence_df[text_column],
    'True_Label': true_values,
    'Predicted_Label': predictions,
    'Correct_Prediction': true_values == predictions
})
results_df.to_csv(os.path.join(results_folder, 'prediction_results.csv'), index=False)

# Save metrics to text file
with open(os.path.join(results_folder, 'metrics.txt'), 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))

# Create some additional visualizations

# 1. Pie chart of true label distribution
plt.figure(figsize=(8, 6))
true_counts = true_values.value_counts()
plt.pie(true_counts, labels=['No (0)', 'Yes (1)'] if 0 in true_counts.index else ['Yes (1)', 'No (0)'], 
        autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
plt.title('Distribution of True Labels')
plt.savefig(os.path.join(results_folder, 'true_label_distribution.png'))
plt.close()

# 2. Bar chart comparing prediction success by class
by_class = results_df.groupby('True_Label')['Correct_Prediction'].mean() * 100
plt.figure(figsize=(8, 6))
by_class.plot(kind='bar', color=['lightblue', 'lightgreen'])
plt.xlabel('True Label')
plt.ylabel('Prediction Accuracy (%)')
plt.title('Prediction Accuracy by Class')
plt.xticks([0, 1], ['No (0)', 'Yes (1)'])
plt.ylim(0, 100)
for i, value in enumerate(by_class):
    plt.text(i, value + 2, f"{value:.1f}%", ha='center')
plt.savefig(os.path.join(results_folder, 'accuracy_by_class.png'))
plt.close()

print(f"All results saved to folder: {results_folder}")



