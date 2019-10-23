### Machine Learning Notes

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


 
