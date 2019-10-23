Notes on Machine Learning and the Frameworks I use.

### Popular ML Libraries
- Scikit-Learn
- TensorFlow
- Keras
- PyTorch
- XGBoost
- LightGBM
- CatBoost


----

## Model Evaluation Metrics

### Regression Evaluation Metrics
- [Sklearn Regression Metrics](https://scikit-learn.org/stable/modules/classes.html#regression-metrics)


-----
### Classification Evaluation Metrics
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


- [Sklearn - Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)


-----


## Feature Scaling

### Feature Scaling Articles
- [Sebastian Raschka](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html)

**How should I normalize features in machine learning assuming logistic regression?**
- I have a continuous variable ranging from 0 to 1,000,000 and some binary variables (0 or 1). What’s the best way to normalize the continuous variable and why?

Logistic regression is linear. Any linear normalization, while useful for speeding up convergence (negligible unless dataset is huge) and for interpreting coefficients, will not change your results in any way.

I am a fan of subtracting the mean and dividing by the standard deviation because of its nice theoretical properties, but there's something else you have to consider before that: the distribution of the data. You usually want it to have a “nice” distribution like a normal or uniform distribution.

Plot a histogram of your continuous variable. If it looks like higher values get exceedingly rare, then it may be a power law, in which case a log Transform is appropriate.

In fact, plot two histograms on the same axes, one for each value of the response variable. If your only other variable is the binary input, plot all four histograms on the same axes. Then plot your data on the axes defined by your explanatory variables with the scatter plot point’s colored by the correct output. It's always good to get a solid visualization of your data. This can tell you if logistic regression is even appropriate. If you can't draw a line between the two colors (with a tolerable amount of exceptions), then logistic regression probably isn't the right tool. You might find a Gaussian Mixture Model is better.

If logistic regression is appropriate, the histograms will help you decide what transformations to use. Once you've done those transformations, I'd recommend subtracting the mean and dividing by the standard deviation, particularly if you're using any regularization.


**Sklearn Documentation - Preprocessing**
https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing


All algorithms that are distance based require scaling. This includes all curve fitting algorithms (linear/non-linear regressions), logistic regression, KNN, SVM, Neural Networks, clustering algorithms like k-means clustering etc.

Algorithms that are used for matrix factorization, decomposition or dimensionality reduction like PCA, SVD, Factorization Machines etc also require normalization.

Algorithms that do not require normalization/scaling are the ones that rely on rules. They would not be affected by any monotonic transformations of the variables. Scaling is a monotonic transformation - the relative order of smaller to larger value in a variable is maintained post the scaling. Examples of algorithms in this category are all the tree based algorithms - CART, Random Forests, Gradient Boosted Decision Trees etc. These algorithms utilize rules (series of inequalities) and do not require normalization.

Also, Algorithms that rely on distributions of the variables, like Naive Bayes also do not need scaling.
- source: https://www.quora.com/Which-machine-algorithms-require-data-scaling-normalization

Why and Where to Apply Feature Scaling?
Real world dataset contains features that highly vary in magnitudes, units, and range. Normalisation should be performed when the scale of a feature is irrelevant or misleading and not should Normalise when the scale is meaningful.

The algorithms which use Euclidean Distance measure are sensitive to Magnitudes. Here feature scaling helps to weigh all the features equally.

Formally, If a feature in the dataset is big in scale compared to others then in algorithms where Euclidean distance is measured this big scaled feature becomes dominating and needs to be normalized.

Examples of Algorithms where Feature Scaling matters
1. K-Means uses the Euclidean distance measure here feature scaling matters.
2. K-Nearest-Neighbours also require feature scaling.
3. Principal Component Analysis (PCA): Tries to get the feature with maximum variance, here too feature scaling is required.
4. Gradient Descent: Calculation speed increase as Theta calculation becomes faster after feature scaling.

Note: Naive Bayes, Linear Discriminant Analysis, and Tree-Based models are not affected by feature scaling.
In Short, any Algorithm which is Not Distance based is Not affected by Feature Scaling.


--
In Algebra, Normalization seems to refer to the dividing of a vector by its length and it transforms your data into a range between 0 and 1

And in statistics, Standardization seems to refer to the subtraction of the mean and then dividing by its SD (standard deviation). Standardization transforms your data such that the resulting distribution has a mean of 0 and a standard deviation of 1.

Regularization is a technique to avoid overfitting when training the machine learning algorithms. The model will have a low accuracy if it is overfitted and to overcome this regularization can be achieved by constraining and regularizing the coefficient estimating it towards zero.


--

You should read this page comp.ai.neural-nets FAQ, Part 2 of 7: LearningSection - Should I normalize/standardize/rescale the
Very informative and nicely explained.

In short,
Z-score normalization is given by
Z=(X−mean(X))/sd(X)

Min-Max Normalization is given by
Z=(X−min(X))/(max(X)−min(X))
where, X is the training data and Z is the normalized training data.

Both the techniques are famous for normalizing data in machine learning/ deep learning.

$$
\frac{n!}{k!(n-k)!} = {n \choose k}
$$


--

I typically use standardization over "normalization" (min-max scaling) since you get mean-centering for free, which is important in certain algorithms, too.
Algorithms where feature scaling matters are

- k-means if you use, for example, Euclidean distance since you typically want all features to contribute equally
- k-nearest neighbors (see k-means)
- logistic regression, SVMs, perceptrons, neural networks etc if you are using gradient descent/ascent-based optimization, otherwise some weights will update much faster than others, for example
- linear discriminant analysis, principal component analysis, kernel principal component analysis since you want to find directions of maximizing the variance (under the constraints that those directions/eigenvectors/principal components are orthogonal); you want to have the same scale here since you'd emphasize variables on "larger measurement scales" more



Data normalization in machine learning is called feature scaling. There are three main methods:

Rescaling (also called min-max scaling)

𝑥𝑛𝑜𝑟𝑚=𝑥−𝑥𝑚𝑖𝑛𝑥𝑚𝑎𝑥−𝑥𝑚𝑖𝑛

The data is transformed to a scale of [0,1].

Standardization

𝑥𝑛𝑜𝑟𝑚=𝑥−𝜇𝜎

The data is normalized to a Z-score, or standard score.

Scaling to unit length

𝑥𝑛𝑜𝑟𝑚=𝑥||𝑥||

where ||𝑥|| is the Euclidian length of the feature vector.

In my opinion, min-max scaling should be avoided if possible, because features with long-tail distributions will be dominated by features with uniform distributions. Standardization mostly solves this problem, but cannot be applied when the data has to fit within exact boundaries, such as with many neural network algorithms.

- https://qr.ae/TWFUjk


------------






### Some insightful ML articles
- [Scikit-Learn: A silver bullet for basic machine learning](https://medium.com/analytics-vidhya/scikit-learn-a-silver-bullet-for-basic-machine-learning-13c7d8b248ee)




## A Tour of Various Machine Learning Algorithms

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

