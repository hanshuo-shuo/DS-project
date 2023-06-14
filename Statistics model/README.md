# GLM

## INTRO 

GLM stands for Generalized Linear Models, which is a flexible generalization of ordinary linear regression that allows for response variables that have error distribution models other than a normal distribution. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

GLM consists of three main components:

- Random Component: This is the probability distribution that is assumed for the response variable. For example, normal distribution is used in linear regression, binomial distribution in logistic regression, and Poisson distribution in Poisson regression.

- Systematic Component: It is the linear predictor. This is a linear function of the explanatory variables, as in linear regression. For example, if you have two explanatory variables X1 and X2, the linear predictor may be β0 + β1X1 + β2X2.

- Link Function: The link function provides the relationship between the systematic component and the mean of the random component. It is a function that transforms the expected value of the response variable to create a linear relationship with the predictor variables. For example, a common link function is the logit function in logistic regression.

## Choose Model

- Logistic Regression (a type of GLM): Logistic regression is used when the outcome variable is binary (i.e., it takes only two values, which usually represent the occurrence or non-occurrence of some outcome event, usually coded as 0 or 1). The predictors can be of any type (continuous, discrete, binary, etc.). It's used in various fields including machine learning, most medical fields, and social sciences.

- Probit Regression model: The probit link function is the cumulative distribution function (CDF) of a standard normal distribution. It translates the linear regression output to a probability that the response variable equals one of the binary outcomes, ensuring that the predicted values fall between 0 and 1.

- Poisson Regression (a type of GLM): This is used when the outcome variable is a count variable (i.e., number of times an event occurs). An important assumption of the Poisson model is that the mean of the distribution is equal to its variance. It's often used in fields such as biology, insurance, medicine, and other fields where count data is common.

- Multinomial Logistic Regression (a type of GLM): This model is used when the outcome variable is categorical with more than two levels. That is, the outcome can have three or more possible types that are not ordered. It's often used in machine learning and fields where categorical outcomes are common.

- Ordinal Logistic Regression (a type of GLM): This model is used when the outcome variable is ordinal, i.e., categories have a specific order (like ratings on a scale from 1 to 5). The predictors can be of any type.

- Negative Binomial Regression (a type of GLM): This is used when the outcome variable is a count and the mean and variance are not equal (i.e., overdispersion is present in the data). It's often used in ecology and other fields where overdispersed count data is common.

- Quasi-Poisson Regression, which is similar to Poisson regression, but it relaxes the assumption that the mean and variance of the distribution are equal. This model is used when there is overdispersion in the count data (i.e., the variance is larger than the mean).
- Inverse Gaussian Regression is another GLM that's used for non-negative continuous data, especially when the variance increases more quickly than the mean. It's less commonly used than the other models, but it can be useful in certain situations.
