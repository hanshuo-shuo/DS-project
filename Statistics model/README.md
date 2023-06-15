# Generalized Linear Models

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

# Nonlinear models

Nonlinear models are used in statistics when the relationship between the dependent and independent variables is not linear. Nonlinear models can represent more complex relationships, and can be used in a wide variety of fields including economics, engineering, biology, and physics.

- Polynomial Regression: This type of regression analysis models the relationship between the independent variable x and the dependent variable y as an nth degree polynomial. It can model relationships where the effect of the predictor variable does not increase or decrease linearly, but can increase at a decreasing rate, or even start to decrease after a certain point.

- Generalized Additive Models (GAMs): These are a flexible class of models that can be used when you suspect that your response variable has a nonlinear relationship with some of your predictors. GAMs allow you to use linear regression where the response is linear in terms of the predictors, but also allow for nonlinear relationships.
- Nonparametric regression : (Like loess, local-linear regression)refers to techniques that model the relationship between variables without making specific assumptions about the form or parameters of the function that relates the variables. This is in contrast to parametric regression techniques, like linear regression, which assumes that the relationship can be modeled by a specific form, such as a straight line.

# Missing Value

## SINGLE IMPUTATION METHODS
- Mean substitution: When a value is missing, substitute that variables mean value. 
- Conditional mean substitution: When a value is missing, predict it using a MR model.
- Hot deck imputation: non-respondents are matched to resembling respondents and the missing value is imputed with the score of that similar respondent
- EM ALGORITHM can be used to do single imputation

## Multiple Imputation
Multiple Imputation is a statistical technique used to handle missing data in a dataset. The idea behind multiple imputation is that, rather than filling in a single value for each missing value, it's better to recognize the uncertainty around what value should be imputed by imputing multiple values and creating multiple "complete" datasets.

Multiple Imputation using Chained Equations (MICE) in R. 

## Truncated
Treat the missing values as truncated at 1.00 and employ Heckmanís selection-regression model;

```{R}
# Load required packages
library(tidyverse)
library(sampleSelection)
# Create a copy of the dataset
long_phd_heck_data <- long_phd
# Create a new column "lfp" and assign 1 if job > 1 else 0
long_phd_heck_data <- long_phd_heck_data %>%
  mutate(lfp = if_else(job > 1, "1", "0"))
# Replace missing values in the "lfp" column with 0
long_phd_heck_data$lfp[is.na(long_phd_heck_data$lfp)] <- 0
# Fit Heckman selection model to the data
heck_long_model <- selection(lfp ~ gender + phd + mentor + fellowship + articles + citations,
                              job ~ gender + phd + mentor + fellowship + articles + citations, data = long_phd_heck_data)
# Display the summary of the Heckman selection model
summary(heck_long_model)
```

## Censored Regression

(c) treat the missing values as censored and fit the Tobit model.

```{r, message=FALSE}
# Load required package
library(censReg)
# Create a copy of the dataset
long_phd_tobit_data <- long_phd
# Replace missing values in the "job" column with 1
long_phd_tobit_data$job[is.na(long_phd_tobit_data$job)] <- 1
# Fit Tobit regression model to the data
tobit_model <- censReg(job ~ gender + phd + mentor + fellowship + articles + citations, left = 1, right = Inf, data = long_phd_tobit_data)
summary(tobit_model)
```



