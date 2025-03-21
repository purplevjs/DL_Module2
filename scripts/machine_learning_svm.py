import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from Data_preprocess import data_preprocess
import joblib
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Create results storage folder
results_dir = "svm_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# 1. Read data
sentence_df = data_preprocess()
print("DataFrame columns:", sentence_df.columns.tolist())


# Print Longstorage column data type and unique values
print("\nLongstorage dtype:", sentence_df["Longstorage"].dtype)
print("Longstorage unique values:", sentence_df["Longstorage"].unique())


# Convert Longstorage column to standard binary classification labels (0/1)
# Convert "Yes"/"yes" to 1, "No"/"no" to 0
def normalize_yes_no(value):
    if isinstance(value, (int, float)):
        return 1 if value == 1 else 0
    elif isinstance(value, str):
        value = value.strip().lower()
        if value in ['yes', 'y']:
            return 1
        elif value in ['no', 'n']:
            return 0
        else:
            # For unrecognized values, return -1 as invalid
            print(f"Warning: Unrecognized value '{value}', set to -1")
            return -1
    else:
        return -1

# Apply conversion function
sentence_df['Longstorage_binary'] = sentence_df['Longstorage'].apply(normalize_yes_no)

# Check for invalid values (-1), and remove if any
invalid_count = (sentence_df['Longstorage_binary'] == -1).sum()
if invalid_count > 0:
    print(f"Found {invalid_count} invalid values, will be removed from dataset")
    sentence_df = sentence_df[sentence_df['Longstorage_binary'] != -1]

print("\nUnique values after conversion to Longstorage_binary:", sentence_df['Longstorage_binary'].unique())
print("Count by category:")
print(sentence_df['Longstorage_binary'].value_counts())

# 2. Text feature extraction
# Use TF-IDF to convert text to numerical vectors
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(sentence_df["Sentence"])  # X shape is (n_samples, n_features)

# 3. Prepare target variables
# Regression task: predict Importance (continuous value)
y_reg_importance = sentence_df["Importance"].values  

# Classification task: predict Longstorage (binary)
y_clf_longstorage = sentence_df["Longstorage_binary"].values

# 4. Split training and test sets - separate splits for each model
X_train_importance, X_test_importance, y_train_importance, y_test_importance = train_test_split(
    X, y_reg_importance, test_size=0.2, random_state=42
)

X_train_longstorage, X_test_longstorage, y_train_longstorage, y_test_longstorage = train_test_split(
    X, y_clf_longstorage, test_size=0.2, random_state=42
)


# =========================================
# Model 1: Support Vector Regression model for Importance
# =========================================
print("\n===== Training Importance Regression Model =====")
svr_importance = SVR(kernel="linear", C=1.0, epsilon=0.1)
svr_importance.fit(X_train_importance, y_train_importance)
y_pred_importance = svr_importance.predict(X_test_importance)

# Calculate regression metrics
mse_importance = mean_squared_error(y_test_importance, y_pred_importance)
r2_importance = r2_score(y_test_importance, y_pred_importance)
print("Importance SVR MSE:", mse_importance)
print("Importance SVR R2:", r2_importance)

# Save regression results
reg_results_importance = pd.DataFrame({
    'True_Values': y_test_importance,
    'Predicted_Values': y_pred_importance,
    'Squared_Error': (y_test_importance - y_pred_importance) ** 2
})
reg_results_importance.to_csv(os.path.join(results_dir, 'importance_regression_results.csv'), index=False)

# Visualize regression results and save
plt.figure(figsize=(10, 6))
plt.scatter(y_test_importance, y_pred_importance, alpha=0.5)
plt.plot([min(y_test_importance), max(y_test_importance)], [min(y_test_importance), max(y_test_importance)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('SVR for Importance: True vs Predicted Values')
plt.savefig(os.path.join(results_dir, 'importance_regression_plot.png'))
plt.close()

# =========================================
# Model 2: Support Vector Classifier for Longstorage (as binary classification)
# =========================================
print("\n===== Training Longstorage Classification Model =====")
svc = SVC(kernel="linear", C=1.0, probability=True)
svc.fit(X_train_longstorage, y_train_longstorage)
y_pred_longstorage = svc.predict(X_test_longstorage)
y_pred_longstorage_prob = svc.predict_proba(X_test_longstorage)  # Get probability predictions

# Calculate classification metrics
acc = accuracy_score(y_test_longstorage, y_pred_longstorage)
report = classification_report(y_test_longstorage, y_pred_longstorage)
print("Longstorage SVC Classification Accuracy:", acc)
print("\nLongstorage Classification Report:")
print(report)

# Save classification results
clf_results = pd.DataFrame({
    'True_Labels': y_test_longstorage,
    'Predicted_Labels': y_pred_longstorage,
    'Probability_Class_0': y_pred_longstorage_prob[:, 0],
    'Probability_Class_1': y_pred_longstorage_prob[:, 1]
})
clf_results.to_csv(os.path.join(results_dir, 'longstorage_classification_results.csv'), index=False)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test_longstorage, y_pred_longstorage_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Longstorage Classification')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'longstorage_roc_curve.png'))
plt.close()

# Plot confusion matrix
cm = confusion_matrix(y_test_longstorage, y_pred_longstorage)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No (0)', 'Yes (1)'],
            yticklabels=['No (0)', 'Yes (1)'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Longstorage Classification')
plt.savefig(os.path.join(results_dir, 'longstorage_confusion_matrix.png'))
plt.close()

# Save model metrics
metrics = {
    'Importance_SVR_MSE': mse_importance,
    'Importance_SVR_R2': r2_importance,
    'Longstorage_SVC_Accuracy': acc
}
with open(os.path.join(results_dir, 'model_metrics.txt'), 'w') as f:
    for metric_name, metric_value in metrics.items():
        f.write(f"{metric_name}: {metric_value}\n")
    f.write("\nLongstorage Classification Report:\n")
    f.write(report)

# Save models and preprocessing information
joblib.dump(svr_importance, os.path.join(results_dir, 'importance_svr_model.pkl'))
joblib.dump(svc, os.path.join(results_dir, 'longstorage_svc_model.pkl'))
joblib.dump(vectorizer, os.path.join(results_dir, 'tfidf_vectorizer.pkl'))

# Create mapping dictionary and save for reference
longstorage_mapping = {}
for val in sentence_df['Longstorage'].unique():
    longstorage_mapping[str(val)] = normalize_yes_no(val)

# Save mapping information to file
mapping_df = pd.DataFrame({
    'Original_Value': list(longstorage_mapping.keys()),
    'Mapped_To': list(longstorage_mapping.values())
})
mapping_df.to_csv(os.path.join(results_dir, 'yes_no_mapping.csv'), index=False)

# Save true value data
print("\n===== Saving Original True Value Data =====")
# Create true values dataframe and save
true_values_df = pd.DataFrame({
    'Importance': sentence_df["Importance"].values,
    'Longstorage': sentence_df["Longstorage"].values,
    'Longstorage_binary': sentence_df["Longstorage_binary"].values
})
true_values_df.to_csv(os.path.join(results_dir, 'true_values_data.csv'), index=False)

# Save training and test set true values
train_importance_df = pd.DataFrame({'True_Importance': y_train_importance})
test_importance_df = pd.DataFrame({'True_Importance': y_test_importance})
train_importance_df.to_csv(os.path.join(results_dir, 'train_importance_true_values.csv'), index=False)
test_importance_df.to_csv(os.path.join(results_dir, 'test_importance_true_values.csv'), index=False)

train_longstorage_df = pd.DataFrame({'True_Longstorage': y_train_longstorage})
test_longstorage_df = pd.DataFrame({'True_Longstorage': y_test_longstorage})
train_longstorage_df.to_csv(os.path.join(results_dir, 'train_longstorage_true_values.csv'), index=False)
test_longstorage_df.to_csv(os.path.join(results_dir, 'test_longstorage_true_values.csv'), index=False)

print(f"All results saved to {results_dir} folder")