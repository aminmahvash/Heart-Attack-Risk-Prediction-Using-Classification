# Heart Attack Risk Prediction Using Classification

This project implements a binary classification system to predict the risk of a heart attack based on clinical parameters. The goal is to compare multiple **classification** algorithms and identify the most reliable model for medical diagnosis.

## 📌 Project Overview

The dataset contains several medical features (age, sex, cholesterol, chest pain type, etc.). We applied four different classification models to predict the `output` (0 = Lower risk, 1 = Higher risk).

## 🛠 Workflow

1. **Data Preprocessing**: Handled categorical and numerical data.
2. **Feature Scaling**: Applied `StandardScaler` to normalize the data, which is crucial for distance-based models like KNN and SVM.
3. **Data Splitting**: 80% Training and 20% Testing sets.
4. **Modeling**: Evaluated KNN, Decision Tree, Logistic Regression, and SVM.

## 📊 Evaluation Metrics

While **Accuracy** provides a general overview, in medical contexts, we focused on the **Confusion Matrix** to minimize **False Negatives** (cases where a patient at risk is predicted as healthy).

### Model Performance Summary

| Algorithm | Test Accuracy | False Negatives (Missed Cases) |
| --- | --- | --- |
| **KNN (K=32)** | **0.9016** | **1** |
| **Logistic Regression** | 0.8688 | 3 |
| **SVM (RBF Kernel)** | 0.8524 | 4 |
| **Decision Tree** | 0.8361 | 4 |

## 🏆 Final Conclusion

The **K-Nearest Neighbors (KNN)** model outperformed the others with an accuracy of **90.16%**. More importantly, it showed the highest sensitivity, missing only one actual heart attack case in the test set.
