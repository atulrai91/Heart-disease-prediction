This dataset predicts the cardiovascular disease by taking multiple factors which contribute to heart diseases.

This dataset includes 11 common features which can contribute to the cardiovascular problems. The different features are Age, Cholestrol,
FastingSugar, blood pressure, chest pain, chest pain during excerices, maximum heart beat rate, etc

ML Model Name	          Accuracy	AUC	    Precision	Recall	  F1	MCC
Logistic Regression	      0.8967	0.9296	0.8973	    0.8967	0.8964	0.7910
Decision Tree	          0.7826	0.8752	0.7823	    0.7826	0.7816	0.5581
kNN	                      0.9022	0.9568	0.9022	    0.9022	0.9022	0.8020
Naive Bayes	              0.8750	0.9223	0.8749	    0.8750	0.8749	0.7468
Random Forest (Ensemble)  0.8587	0.9203	0.8587	    0.8587	0.8583	0.7133
XGBoost (Ensemble)	      0.8696	0.9246	0.8696	    0.8696	0.8696	0.7360

ML Model Name                 Observation about model performance
Logistic Regression           Data has near-linear decision boundaries.High AUC shows strong prediction.Model is simple and effective
Decision Tree                 Has lowest performance with lowest MCC score mainly due to overfitting
kNN                           Highest accuracy with high AUC score and having most balanced predictions
Naive Bayes                   Less accurate than kNN and Logistic regression, stable but slightly lower score
Random Forest (Ensemble)      Slightly lower then kNN and Logistic regression, ensemble learning improved stability.
XGBoost (Ensemble)            It has balanced performance, slightly lower than kNN, good predicitive stability