import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_samples = 500
data = {
    'CurrentRatio': np.random.uniform(0.5, 3, num_samples),
    'DebtEquityRatio': np.random.uniform(0.1, 2, num_samples),
    'ProfitMargin': np.random.uniform(-0.1, 0.3, num_samples),
    'ROA': np.random.uniform(-0.1, 0.2, num_samples),
    'Bankrupt': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]) # 20% bankruptcy rate
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# (In a real-world scenario, this would involve handling missing values, outliers, etc.)
# For this synthetic data, no cleaning is explicitly needed.
# --- 3. Model Training ---
X = df.drop('Bankrupt', axis=1)
y = df['Bankrupt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000) # Increased max_iter to ensure convergence
model.fit(X_train, y_train)
# --- 4. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Bankrupt', 'Bankrupt'])
plt.yticks(tick_marks, ['Not Bankrupt', 'Bankrupt'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
# Save the plot to a file
output_filename = 'confusion_matrix.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")