# GLM

GLM stands for Generalized Linear Models, which is a flexible generalization of ordinary linear regression that allows for response variables that have error distribution models other than a normal distribution. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

GLM consists of three main components:

- Random Component: This is the probability distribution that is assumed for the response variable. For example, normal distribution is used in linear regression, binomial distribution in logistic regression, and Poisson distribution in Poisson regression.

- Systematic Component: It is the linear predictor. This is a linear function of the explanatory variables, as in linear regression. For example, if you have two explanatory variables X1 and X2, the linear predictor may be β0 + β1X1 + β2X2.

- Link Function: The link function provides the relationship between the systematic component and the mean of the random component. It is a function that transforms the expected value of the response variable to create a linear relationship with the predictor variables. For example, a common link function is the logit function in logistic regression.
