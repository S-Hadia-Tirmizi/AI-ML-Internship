import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error
bank_data = pd.read_csv('C:\\Users\\Lenovo X230 Tablet\\OneDrive\\Desktop\\bank.csv')
print(bank_data.head())
print(bank_data.isnull().sum())
label_encoder = LabelEncoder()
for col in bank_data.select_dtypes(include='object').columns:
    if col != 'y':
        bank_data[col] = label_encoder.fit_transform(bank_data[col])
bank_data['y'] = bank_data['y'].map({'yes': 1, 'no': 0})
X = bank_data.drop('y', axis=1)
y = bank_data['y']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)
selected_cols = X.columns[selector.get_support()]
print("Top 10 selected features:", selected_cols.tolist())
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='f1')
    print(f"{name} F1 Score (5-fold): {np.mean(scores):.4f}")