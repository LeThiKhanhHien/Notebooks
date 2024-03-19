# Credit Card Fraud Detection # 

Author: Hien Le

<img src="https://securityintelligence.com/wp-content/webp-express/webp-images/doc-root/wp-content/uploads/2016/09/credit-card-fraud-remains-a-risk.jpg.webp" alt="Image Description" style="width: 300px;"/>

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

## Data preparation and observation ## 

1. The data has the shape (284807, 31); there are 29 feature columns.
2. There are no nan values in the data
3. The dataset is imbalanced
   
   ![image](https://github.com/LeThiKhanhHien/Notebooks/assets/56401620/43902dd9-f987-4ef2-9977-663b6fa97761)

   ![image](https://github.com/LeThiKhanhHien/Notebooks/assets/56401620/c7fbad7a-d125-4822-99e3-70b7f13870d5)


## Q1: How can we compare the performance of different models? ##

In essence, Credit Card Fraud Detection involves identifying anomalies within credit card transactions, making it an anomaly detection task. When evaluating the performance of anomaly detection models, we typically rely on the following criteria. Note that positive = anomaly, negative = normality. 

|          | Predicted Negative | Predicted Positive |
|----------|--------------------|--------------------|
| Actual Negative | TN               | FP               |
| Actual Positive | FN               | TP               |


- **Precision and Recall**: Precision represents the proportion of correctly identified anomalies among all instances labeled as anomalies by the model, while recall measures the proportion of true anomalies that are correctly identified by the model.
  $$Precision= \frac{TP}{TP+FP}$$
  $$Recall= \frac{TP}{TP + FN} $$ 

- **F1 Score**: The F1 score is the harmonic mean of precision and recall and provides a balanced measure of a model's performance.
$$F1 Score=2 \times \frac{Precision×Recall}{Precision+Recall}$$

- **Area Under the ROC Curve (AUC-ROC)**: The ROC curve plots the true positive rate against the false positive rate at various threshold settings. AUC-ROC provides a single scalar value that represents the overall performance of the model across different threshold settings.

- **Area Under the Precision-Recall Curve (AUC-PR)**: Similar to AUC-ROC, AUC-PR summarizes the precision-recall curve's performance, particularly useful when dealing with imbalanced datasets.

## Q2. Can we say somethings about the importance level of the features in the dataset?

In fact, the feature importance depends much on the model we use. 

Let us first use Ordinary Least Squares (OLS) regression from the statsmodels library to fit a linear regression model and obtain p-values for each feature coefficient. These p-values indicate the statistical significance of each feature in predicting the target variable. **In the context of linear regression**, if the p-value is less than a chosen significance level (commonly 0.05), it suggests that the corresponding feature has a statistically significant relationship with the target variable; a high p-value suggests that the corresponding feature may not have a statistically significant relationship with the target variable. Features with low p-values are likely to be important predictors, while features with high p-values may be less relevant and could potentially be removed from the model. **Note that using p-value is sometimes controversial.**

```python
#from statsmodels.regression.linear_model import OLS
#model = OLS(y, X_scaled).fit()
#pvalues = pd.DataFrame(model.pvalues)
#pvalues.reset_index(inplace=True)
#pvalues.rename(columns={0: "pvalue", "index": "feature"}, inplace=True)
#pvalues.style.background_gradient(cmap='RdYlGn')

