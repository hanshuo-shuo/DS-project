# Credit Card Fraud Data Wrangling and EDA

## Dataset

[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/isaikumar/creditcardfraud)

## Analysis

1. Perform preliminary data quality checks, such as identifying duplicated columns and columns with entirely missing data. Determine how to manage these issues and justify your approach for handling them. 

2. Pay close attention to outliers in numerical variables. Describe the methods you use for detecting outliers and explain your chosen approach for handling them. Justify your decisions and explain the potential impact of outliers on the analysis. 

3. Identify columns with missing values and determine how to manage them. Justify your approach and reasoning for handling missing values in the dataset.

4. Investigate the time variables in the dataset and address any potential issues that may arise when working with them. This may involve converting the variables to a suitable format, conducting additional cleaning, and/or extracting meaningful features to ensure consistency and usability in the analysis. Justify your approach and reasoning for handling time variables, explaining how your decisions enhance the overall data quality and interpretation.

5. Certain columns in the dataset may require special treatment during data wrangling due to their unique characteristics (e.g., `cardCVV`, `enteredCVV`, `cardLast4Digits`). Explore alternative methods for integrating these variables into your analysis, and document any decisions made during this stage.

6. Analyze the relationship between the columns `cardCVV`, `enteredCVV`, and `cardLast4Digits` and the target variable, `isFraud`, using an appropriate visualization (such as a grouped bar chart). Discuss the insights gained about the relationship between these variables and credit card fraud. 

7. Visualize the distribution of `transactionAmount` using an appropriate plot, such as a histogram or density plot. Provide a brief analysis of the observed pattern and discuss any insights or trends you can infer from the visualization.

8. Investigate the relationship between `isFraud` and categorical predictors, such as `merchantCategoryCode`, `posEntryMode`, `transactionType`, `posConditionCode`, and `merchantCountryCode`, by creating suitable visualizations like bar charts to display the fraud rate for each category. Describe the patterns you observe and their potential implications for creating a predictive model for fraudulent transactions.

9. Further explore the relationship between `isFraud` and `transactionType` conditioned on `merchantCategoryCode` by generating a grouped bar chart or another suitable visualization to display the fraudulent rates by merchant category code and transaction type. Share any additional insights you have.

10. Construct conditional probability density plots (or other suitable visualizations) for the numerical variables in the dataset to help understand the relationships between these variables and the target variable, `isFraud`. Identify any patterns or trends suggesting a relationship between the numerical variables and fraudulent transactions.

11. Programmatically identify multi-swipe transactions by defining specific conditions under which they occur (e.g., same amount, within a short time span, etc.). Clearly state the conditions you have chosen for this analysis. Estimate the percentage of multi-swipe transactions and the percentage of the total dollar amount for these transactions, excluding the first "normal" transaction from the count. Discuss any interesting findings or patterns that emerge from your analysis of multi-swipe transactions and their conditions.

12. Examine the class imbalance in the `isFraud` outcome variable and discuss the potential implications of these patterns for the development of a predictive model for credit card fraud detection. Note that at this stage, we are not building or training a predictive model. Instead, our objective is to gain a deeper understanding of the class imbalance issue in the data and explore ways to address it.

13. Implement a method of your choice to mitigate class imbalance in the isFraud outcome variable. Describe the method you used and report its effects on the class distribution. How might addressing class imbalance impact the effectiveness and performance of a predictive model for credit card fraud detection?
