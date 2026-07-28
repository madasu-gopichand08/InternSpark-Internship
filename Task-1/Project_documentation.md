# Heart Disease Prediction using Machine Learning

# 1. Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction using Machine Learning can help doctors identify patients who are at risk and support timely medical decisions.

This project uses machine learning algorithms to predict whether a patient is likely to have heart disease based on various medical attributes.

---

# 2. Problem Statement

The objective of this project is to build a classification model that predicts the presence of heart disease using patient health information.

---

# 3. Objectives

- Analyze the heart disease dataset.
- Perform data preprocessing.
- Train multiple Machine Learning models.
- Compare model performance.
- Select the best-performing model.

---

# 4. Dataset

Dataset: UCI Heart Disease Dataset

The dataset contains medical attributes such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- Slope
- Number of Major Vessels
- Thalassemia

Target Variable:

- 0 → No Heart Disease
- 1 → Heart Disease

---

# 5. Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset inspection
- Missing value check
- Duplicate value check
- Statistical summary
- Target class distribution
- Histograms
- Boxplots
- Scatter plots
- Correlation heatmap

EDA helped understand feature distributions and relationships within the dataset.

---

# 6. Data Preprocessing

The preprocessing steps included:

- Splitting features and target variable
- Train-test split (80:20)
- Feature scaling using StandardScaler (for Logistic Regression)

Random Forest was trained on the original feature values because tree-based algorithms do not require feature scaling.

---

# 7. Machine Learning Models

## Logistic Regression

A linear classification algorithm suitable for binary classification problems.

## Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy.

---

# 8. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Cross Validation

---

# 9. Results

| Model | Accuracy | ROC-AUC |
|--------|----------|---------|
| Logistic Regression | 85% | 0.93 |
| Random Forest | 84% | 0.92025 |

Logistic Regression achieved slightly better performance than Random Forest on this dataset.

---

# 10. Conclusion

Both models performed well in predicting heart disease.

Logistic Regression achieved the best overall performance with an ROC-AUC score of 0.93, making it the preferred model for this project.

---

# 11. Future Improvements

- Hyperparameter tuning
- Feature selection
- Testing additional algorithms such as XGBoost
- Deploying the model using Flask or Streamlit
- Training on a larger dataset

---

# 12. References

- UCI Machine Learning Repository
- Scikit-learn Documentation
- Pandas Documentation
- NumPy Documentation
