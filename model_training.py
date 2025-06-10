import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import pickle

print("Loading dataset...")
df = pd.read_csv('data/train.csv.gz', compression='gzip')

print("Sampling 1% of data for faster training...")
df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)

print("Dropping 'id' column...")
df = df.drop(columns=['id'])

print("Dropping unused and high-cardinality columns...")
# Drop columns that cause memory issues or aren't useful
high_card_cols = ['device_ip', 'site_id', 'app_id']
df = df.drop(columns=high_card_cols)

print("Separating features and target...")
X = df.drop('click', axis=1)
y = df['click']

# Define numerical and categorical columns
numerical_cols = ['banner_pos', 'device_type', 'device_conn_type']
categorical_cols = [col for col in X.columns if col not in numerical_cols]

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Preprocessing pipeline (use sparse matrix)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True)
, categorical_cols),
    ],
    remainder='passthrough'
)

# XGBoost pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3))
])

print("Splitting data into train and test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
pipeline.fit(X_train, y_train)
print("Training completed.")

print("Evaluating model...")
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")

print("Saving model to 'xgb_pipeline.pkl'...")
with open('xgb_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# === Plotting ===

print("Plotting ROC curve...")
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.4f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()

print("Plotting Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("Plotting Precision-Recall Curve...")
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.tight_layout()
plt.show()

print("All done!")
