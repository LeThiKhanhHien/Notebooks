# Credit Card Fraud Detection #
Author: Hien Le

## Context ## 
Credit card companies must possess the capability to detect fraudulent credit card transactions to ensure customers aren't billed for unauthorized purchases.

Three questions we will answer in this notebook
1. How can we compare the performance of different models?
2. Can we say somethings about the importance level of the features in the dataset? 
3. Explore some models and which one among them is the best for credit card fraud detection task? 

## The dataset ##
The dataset is downloaded from [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

The dataset comprises credit card transactions made by European cardholders in September 2013. Within a span of two days, there were 492 fraudulent transactions out of a total of 284,807 transactions. This dataset is characterized by significant imbalance, with the positive class (frauds) representing only 0.172% of all transactions.

The dataset includes solely numerical input variables resulting from a PCA transformation. Due to confidentiality constraints, the original features or additional background information about the data are not provided. Principal component analysis (PCA) was applied to derive features V1 through V28; however, features 'Time' and 'Amount' were not subjected to PCA transformation. 'Time' denotes the elapsed seconds between each transaction and the first recorded transaction in the dataset, while 'Amount' represents the transaction amount. The 'Class' feature serves as the response variable, taking a value of 1 in cases of fraud and 0 otherwise.