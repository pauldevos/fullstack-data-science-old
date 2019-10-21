### Machine Learning Notes



*How should I normalize features in machine learning assuming logistic regression?*
- I have a continuous variable ranging from 0 to 1,000,000 and some binary variables (0 or 1). What’s the best way to normalize the continuous variable and why?
```
Logistic regression is linear. Any linear normalization, while useful for speeding up convergence (negligible unless dataset is huge) and for interpreting coefficients, will not change your results in any way.

I am a fan of subtracting the mean and dividing by the standard deviation because of its nice theoretical properties, but there's something else you have to consider before that: the distribution of the data. You usually want it to have a “nice” distribution like a normal or uniform distribution.

Plot a histogram of your continuous variable. If it looks like higher values get exceedingly rare, then it may be a power law, in which case a log Transform is appropriate.

In fact, plot two histograms on the same axes, one for each value of the response variable. If your only other variable is the binary input, plot all four histograms on the same axes. Then plot your data on the axes defined by your explanatory variables with the scatter plot point’s colored by the correct output. It's always good to get a solid visualization of your data. This can tell you if logistic regression is even appropriate. If you can't draw a line between the two colors (with a tolerable amount of exceptions), then logistic regression probably isn't the right tool. You might find a Gaussian Mixture Model is better.

If logistic regression is appropriate, the histograms will help you decide what transformations to use. Once you've done those transformations, I'd recommend subtracting the mean and dividing by the standard deviation, particularly if you're using any regularization.
```



