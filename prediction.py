# ---------------------------
# 1. Import Libraries
# ---------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ---------------------------
# 2. Load Dataset
# ---------------------------

df = pd.read_csv("manufacturing_defects_large.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------
# 3. Exploratory Data Analysis
# ---------------------------

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

sns.countplot(x='defect', data=df)
plt.title("Defect Distribution")
plt.show()

# ---------------------------
# 4. Split Data (Features + Target)
# ---------------------------

X = df.drop("defect", axis=1)
y = df["defect"]

# ---------------------------
# 5. Feature Scaling
# ---------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 6. Train-Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------
# 7. Model Training (Random Forest)
# ---------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# ---------------------------
# 8. Predictions
# ---------------------------

y_pred = model.predict(X_test)

# ---------------------------
# 9. Evaluation
# ---------------------------

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# 10. Feature Importance
# ---------------------------

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10,5))
feature_importance.plot(kind='bar')
plt.title("Feature Importance")
plt.show()
