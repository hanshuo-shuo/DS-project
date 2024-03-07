# Multilevel Modeling

MLM is an acronym for Multilevel Modeling, also known as hierarchical linear modeling (HLM), mixed-effect modeling, or random coefficient modeling. These statistical models are used to analyze data that has a hierarchical or nested structure. For example, consider students nested within classrooms, which are nested within schools. In such cases, it's likely that students in the same classroom are more similar to each other than to students from different classrooms. Similarly, classrooms in the same school are more likely to be similar to each other than to classrooms from different schools. Multilevel models can take this clustering of data into account, allowing for residual components at each level in the data hierarchy.

Fitting the linear mixed-effects model:
`model <- lmer(test ~ SES_c + IQ_c + (1 + SES_c + IQ_c | school), data = data)`

FITTING generalized linear mixed-effects model:
```{R}
model <- glmer(high_pass ~ SES_c + IQ_c + (1 + SES_c + IQ_c | school), family = binomial, data = data)
S(model)
```

Here, test is the outcome variable (possibly a test score), and SES_c and IQ_c are predictor variables (possibly socio-economic status and IQ score, centered). The 1 in the formula denotes the intercept. The | school indicates that the intercept and slopes for SES_c and IQ_c are allowed to vary across different schools, hence the mixed-effects model.

# Longitudinal Data Modeling

 - Mixed-Effects Models: Mixed-effects models (also known as multilevel models or hierarchical models) are a common choice for longitudinal data. They can account for the correlation between repeated measures on the same units by including random effects for the units. The lme4 and nlme packages in R can be used to fit mixed-effects models.
- Generalized Estimating Equations (GEE): GEE models are another common choice for longitudinal data. GEE models are similar to mixed-effects models but they focus on estimating the average response over the population rather than the response for a specific individual. GEE models can be fitted in R using the geepack package.
- Growth Curve Models: Growth curve models, which are a special type of mixed-effects model, are often used for longitudinal data. These models include fixed and random effects for time (or age), which allows for modeling individual growth trajectories. Growth curve models can be fitted using the nlme or lme4 packages.
- Survival Analysis: If the longitudinal data includes time-to-event outcomes, survival analysis methods can be used. The survival package in R provides a variety of functions for survival analysis.
```{R}
library(survival)
model <- coxph(Surv(time, event) ~ x, data = data)
```

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

Treat the missing values as censored and fit the Tobit model.

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
# Model Selection

- AIC (Akaike's Information Criterion) and BIC (Bayesian Information Criterion): These are both commonly used criteria for model selection. Lower AIC or BIC values indicate better fitting models. AIC and BIC are available for many models through the AIC() and BIC() functions in R.

- Stepwise Regression: This method starts with a full or null model, and tries to add or remove predictors based on some criteria, such as the AIC or BIC. The step() function in R can be used for stepwise regression.

- Cross-Validation: Cross-validation is a resampling method used to evaluate machine learning models on a limited data sample. The idea behind it is to divide the data into 'folds', train the model on a subset of the folds and test it on the remaining fold. Then, this is repeated so that each fold acts as the test set once. Model performance can be assessed using various metrics such as Mean Squared Error (MSE) for regression tasks or accuracy for classification tasks. R packages like caret and mlr provide functionality for cross-validation.

- LASSO and Ridge Regression: These are techniques used for creating parsimonious models in the presence of a 'large' number of features. They work by penalizing the absolute size of the regression coefficients, which can lead to some coefficients being exactly zero. This effectively performs feature selection. These methods are implemented in the glmnet package in R.

- Model selection based on the likelihood-ratio test: In R, this can be done using the anova() function, which compares the goodness of fit of two models to see if the more complicated one is significantly better than the simpler one.

- Validation or Test set approach: In this method, the data is split into two parts: a training set and a validation (or test) set. The model is trained on the training set and its performance is evaluated on the validation set. The model with the best performance on the validation set is chosen. The function createDataPartition() from the caret package can be used to split the data.

# reference
Class and projects by Prof. Elizabeth Tipton from Northwestern.
