# Predict duration of taxi trips in New York City

For our final machine learning project, we used the data from a [Kaggle competition](https://www.kaggle.com/c/nyc-taxi-trip-duration)---after preprocessing it---to test the ML algorithms that we coded by ourselves. We made four experiments in total, two per ML algorithm: elastic nets with different optimization approaches, and decision trees with different split criterion.

1. Elastic nets with Stochastic Gradient Descend
2. Elastic nets with Batch Simulated Annealing (our innovation)
3. Decision tree with MSE split criterion
4. Decision tree with MAE split criterion

For the elastic nets, got very close results to the [Scikit-learn implementation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor). However, we couldn't achieve the same with our decision trees.

More details can be found in the ![PDF report](https://github.com/arodriguezca/ny-taxi_trip-duration/blob/master/Taxi-SL-Final-Paper.pdf).

Coded by Alexander Rodriguez and Santosh Malgireddy for the course Machine Learning at the University of Oklahoma.