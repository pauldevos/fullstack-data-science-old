### Popular ML Libraries
- Scikit-Learn
- TensorFlow
- Keras
- PyTorch
- XGBoost
- LightGBM
- CatBoost

### Model Validation Libraries


### Regression Evaluation Metrics


-----

## Classification Evalutation Metrics
Sources:
- [Evaluating a Classification Model: ROC, AUC, Confusion Matric, and Metrics](https://www.ritchieng.com/machine-learning-evaluate-classification-model/)

Topics
1. Review of model evaluation
1. Model evaluation procedures
1. Model evaluation metrics
1. Classification accuracy
1. Confusion matrix
1. Metrics computed from a confusion matrix
1. Adjusting the classification threshold
1. Receiver Operating Characteristic (ROC) Curves
1. Area Under the Curve (AUC)
1. Confusion Matrix Resources
1. ROC and AUC Resources
1. Other Resources

- Confusion Matrix
  - sklearn.metrics.confusion_matrix(y_test, y_pred_class)
  - Accuracy Score
    - sklearn.metrics.accuracy_score(y_test, y_pred_class)
  - Classification Error
    - (1 - sklearn.metrics.accuracy_score(y_test, y_pred_class)
  - Sensitivity = TP / float(FN + TP)
    - sklearn.metrics.recall_score(y_test, y_pred_class)
  - Specificity = TN / (TN + FP)
    - specificity = TN / (TN + FP)
  - false_positive_rate = FP / float(TN + FP)



### Some insightful ML articles
- [Scikit-Learn: A silver bullet for basic machine learning](https://medium.com/analytics-vidhya/scikit-learn-a-silver-bullet-for-basic-machine-learning-13c7d8b248ee)
