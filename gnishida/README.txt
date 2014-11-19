1. How to run?

$ python GradientDescent.py GD <maxIterations> <regularization> <stepSize> <lambda> <featureSet>

The program runs gradient descent algorithm using the training data, and compute the performance on the training, validation, and test dataset. Finally, the results are output.

There are some other options supported as follows:

$ python GradientDescent.py GD <maxIterations> <regularization> <stepSize> <lambda> <featureSet> <draw_graph=Y/N>
  By adding Y as the last argument, the graph of decreasing the objective will be shown.

$ python GradientDescent.py FH <maxIterations> <regularization> <featureSet>
  By using several combination of stepSize and lambda, it will find the best values for the hyperparameters. This will take a long time especially when you specify featureSet=2 or 3.

$ python GradientDescent.py CP <maxIterations> <stepSize> <lambda> <featureSet>
  This will compare the resulting weight vectors between l1 and l2 regularizers.

