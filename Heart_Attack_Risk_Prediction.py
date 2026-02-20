import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics

# 1. Load Data
try:
    df = pd.read_csv('Data/HeartAttackRisk_Data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found in 'Data/' folder.")
    exit()

# 2. Preprocessing
# Selecting features and target
x = np.asarray(df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'caa']])
y = np.asarray(df['output'])

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=34)

# Normalization (StandardScaler)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(f"Train set: {x_train.shape}, Test set: {x_test.shape}")

# ---------------------------------------------------------
# 3. Modeling
# ---------------------------------------------------------

results = {}

# --- Model 1: KNN ---
# Using K=26 as identified in your analysis
knn = KNeighborsClassifier(n_neighbors=26).fit(x_train, y_train)
yhat_knn = knn.predict(x_test)
results['KNN'] = {
    'Train_Acc': metrics.accuracy_score(y_train, knn.predict(x_train)),
    'Test_Acc': metrics.accuracy_score(y_test, yhat_knn)
}

# --- Model 2: Decision Tree ---
dt = DecisionTreeClassifier(criterion="entropy", max_depth=4).fit(x_train, y_train)
yhat_dt = dt.predict(x_test)
results['Decision Tree'] = {
    'Train_Acc': metrics.accuracy_score(y_train, dt.predict(x_train)),
    'Test_Acc': metrics.accuracy_score(y_test, yhat_dt)
}

# --- Model 3: Logistic Regression ---
lr = LogisticRegression(C=0.1, solver='liblinear').fit(x_train, y_train)
yhat_lr = lr.predict(x_test)
results['Logistic Regression'] = {
    'Train_Acc': metrics.accuracy_score(y_train, lr.predict(x_train)),
    'Test_Acc': metrics.accuracy_score(y_test, yhat_lr)
}

# --- Model 4: SVM ---
clf = svm.SVC(kernel='rbf').fit(x_train, y_train)
yhat_svm = clf.predict(x_test)
results['SVM'] = {
    'Train_Acc': metrics.accuracy_score(y_train, clf.predict(x_train)),
    'Test_Acc': metrics.accuracy_score(y_test, yhat_svm)
}

# ---------------------------------------------------------
# 4. Final Evaluation & Comparison Table
# ---------------------------------------------------------

print("\n" + "="*45)
print(f"{'Algorithm':<20} | {'Train Acc':<10} | {'Test Acc':<10}")
print("-" * 45)
for model, acc in results.items():
    print(f"{model:<20} | {acc['Train_Acc']: <10.4f} | {acc['Test_Acc']: <10.4f}")
print("="*45)

# 5. Visualizing the Best Model (KNN) Confusion Matrix
plt.figure(figsize=(6, 5))
cm = metrics.confusion_matrix(y_test, yhat_knn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Risk'], yticklabels=['Healthy', 'Risk'])
plt.title('Best Model: KNN Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()