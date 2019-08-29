### Popular ML Libraries
- Scikit-Learn
- TensorFlow
- Keras
- PyTorch
- XGBoost
- LightGBM
- CatBoost

### Model Validation Libraries
- [Regression Evaluation Metrics](https://github.com/pauldevos/fullstack-data-science/blob/master/machine-learning/README.md#regression-evaluation-metrics)
- [Classification Evalutation Metrics](https://github.com/pauldevos/fullstack-data-science/blob/master/machine-learning/README.md#classification-evaluation-metrics)

----
```<a name="regression-metrics"></a>```
## Regression Evaluation Metrics
- [Sklearn Regression Metrics](https://scikit-learn.org/stable/modules/classes.html#regression-metrics)


-----
```<a name="classification-metrics"></a>```
## Classification Evaluation Metrics
Sources:
- [Evaluating a Classification Model: ROC, AUC, Confusion Matric, and Metrics](https://www.ritchieng.com/machine-learning-evaluate-classification-model/)

- [Sklearn Classification Metrics](https://scikit-learn.org/stable/modules/classes.html#classification-metrics)


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

-----
- [Sklearn - Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)






### Some insightful ML articles
- [Scikit-Learn: A silver bullet for basic machine learning](https://medium.com/analytics-vidhya/scikit-learn-a-silver-bullet-for-basic-machine-learning-13c7d8b248ee)





1. Regression Algorithms
- Ordinary Least Squares Regression (OLSR)
- Linear Regression
- Logistic Regression
- Stepwise Regression
- Multivariate Adaptive Regression Splines (MARS)
- Locally Estimated Scatterplot Smoothing (LOESS)
2. Instance-based Algorithms
- k-Nearest Neighbour (kNN)
- Learning Vector Quantization (LVQ)
- Self-Organizing Map (SOM)
- Locally Weighted Learning (LWL)
3. Regularization Algorithms
- Ridge Regression
- Least Absolute Shrinkage and Selection Operator (LASSO)
- Elastic Net
- Least-Angle Regression (LARS)
4. Decision Tree Algorithms
- Classification and Regression Tree (CART)
- Iterative Dichotomiser 3 (ID3)
- C4.5 and C5.0 (different versions of a powerful approach)
- Chi-squared Automatic Interaction Detection (CHAID)
- Decision Stump
- M5
- Conditional Decision Trees
5. Bayesian Algorithms
- Naive Bayes
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Averaged One-Dependence Estimators (AODE)
- Bayesian Belief Network (BBN)
- Bayesian Network (BN)
6. Clustering Algorithms
- k-Means
- k-Medians
- Expectation Maximisation (EM)
- Hierarchical Clustering
7. Association Rule Learning Algorithms
- Apriori algorithm
- Eclat algorithm
8. Artificial Neural Network Algorithms
- Perceptron
- Back-Propagation
- Hopfield Network
- Radial Basis Function Network (RBFN)
9. Deep Learning Algorithms
- Deep Boltzmann Machine (DBM)
- Deep Belief Networks (DBN)
- Convolutional Neural Network (CNN)
- Stacked Auto-Encoders
10. Dimensionality Reduction Algorithms
- Principal Component Analysis (PCA)
- Principal Component Regression (PCR)
- Partial Least Squares Regression (PLSR)
- Sammon Mapping
- Multidimensional Scaling (MDS)
- Projection Pursuit
- Linear Discriminant Analysis (LDA)
- Mixture Discriminant Analysis (MDA)
- Quadratic Discriminant Analysis (QDA)
- Flexible Discriminant Analysis (FDA)
11. Ensemble Algorithms
- Boosting
- Bootstrapped Aggregation (Bagging)
- AdaBoost
- Stacked Generalization (blending)
- Gradient Boosting Machines (GBM)
- Gradient Boosted Regression Trees (GBRT)
- Random Forest
12. Other Algorithms
- Computational intelligence (evolutionary algorithms, etc.)
- Computer Vision (CV)
- Natural Language Processing (NLP)
- Recommender Systems
- Reinforcement Learning
- Graphical Models


