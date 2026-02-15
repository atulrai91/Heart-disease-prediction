# Problem Statement
This dataset predicts the cardiovascular disease by taking multiple factors which contribute to heart diseases.

# Description
This dataset includes 11 common features which can contribute to the cardiovascular problems. The different features are Age, Cholestrol,
FastingSugar, blood pressure, chest pain, chest pain during excerices, maximum heart beat rate, etc

# Model Performance Comparison

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC   |
|--------------------------|----------|--------|-----------|--------|----------|-------|
| Logistic Regression      | 0.8967   | 0.9296 | 0.8973    | 0.8967 | 0.8964   | 0.7910 |
| Decision Tree            | 0.7826   | 0.8752 | 0.7823    | 0.7826 | 0.7816   | 0.5581 |
| kNN                      | 0.9022   | 0.9568 | 0.9022    | 0.9022 | 0.9022   | 0.8020 |
| Naive Bayes              | 0.8750   | 0.9223 | 0.8749    | 0.8750 | 0.8749   | 0.7468 |
| Random Forest (Ensemble) | 0.8587   | 0.9203 | 0.8587    | 0.8587 | 0.8583   | 0.7133 |
| XGBoost (Ensemble)       | 0.8696   | 0.9246 | 0.8696    | 0.8696 | 0.8696   | 0.7360 |


# Performance Observations

| ML Model Name            | Observation about Model Performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | Data has near-linear decision boundaries. High AUC shows strong prediction capability. Model is simple and effective. |
| Decision Tree            | Has the lowest performance with the lowest MCC score, mainly due to overfitting. |
| kNN                      | Achieved the highest accuracy with a high AUC score and the most balanced predictions. |
| Naive Bayes              | Less accurate than kNN and Logistic Regression; stable but with slightly lower scores. |
| Random Forest (Ensemble) | Slightly lower performance than kNN and Logistic Regression, but ensemble learning improved stability. |
| XGBoost (Ensemble)       | Balanced performance, slightly lower than kNN, with good predictive stability. |
