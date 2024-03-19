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
from statsmodels.regression.linear_model import OLS
model = OLS(y, X_scaled).fit()
pvalues = pd.DataFrame(model.pvalues)
pvalues.reset_index(inplace=True)
pvalues.rename(columns={0: "pvalue", "index": "feature"}, inplace=True)
pvalues.style.background_gradient(cmap='RdYlGn')
```
<table id="T_20da2">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_20da2_level0_col0" class="col_heading level0 col0" >feature</th>
      <th id="T_20da2_level0_col1" class="col_heading level0 col1" >pvalue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_20da2_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_20da2_row0_col0" class="data row0 col0" >x1</td>
      <td id="T_20da2_row0_col1" class="data row0 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_20da2_row1_col0" class="data row1 col0" >x2</td>
      <td id="T_20da2_row1_col1" class="data row1 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_20da2_row2_col0" class="data row2 col0" >x3</td>
      <td id="T_20da2_row2_col1" class="data row2 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_20da2_row3_col0" class="data row3 col0" >x4</td>
      <td id="T_20da2_row3_col1" class="data row3 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_20da2_row4_col0" class="data row4 col0" >x5</td>
      <td id="T_20da2_row4_col1" class="data row4 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_20da2_row5_col0" class="data row5 col0" >x6</td>
      <td id="T_20da2_row5_col1" class="data row5 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_20da2_row6_col0" class="data row6 col0" >x7</td>
      <td id="T_20da2_row6_col1" class="data row6 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_20da2_row7_col0" class="data row7 col0" >x8</td>
      <td id="T_20da2_row7_col1" class="data row7 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_20da2_row8_col0" class="data row8 col0" >x9</td>
      <td id="T_20da2_row8_col1" class="data row8 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_20da2_row9_col0" class="data row9 col0" >x10</td>
      <td id="T_20da2_row9_col1" class="data row9 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_20da2_row10_col0" class="data row10 col0" >x11</td>
      <td id="T_20da2_row10_col1" class="data row10 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_20da2_row11_col0" class="data row11 col0" >x12</td>
      <td id="T_20da2_row11_col1" class="data row11 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_20da2_row12_col0" class="data row12 col0" >x13</td>
      <td id="T_20da2_row12_col1" class="data row12 col1" >0.000220</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_20da2_row13_col0" class="data row13 col0" >x14</td>
      <td id="T_20da2_row13_col1" class="data row13 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_20da2_row14_col0" class="data row14 col0" >x15</td>
      <td id="T_20da2_row14_col1" class="data row14 col1" >0.001614</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_20da2_row15_col0" class="data row15 col0" >x16</td>
      <td id="T_20da2_row15_col1" class="data row15 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_20da2_row16_col0" class="data row16 col0" >x17</td>
      <td id="T_20da2_row16_col1" class="data row16 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_20da2_row17_col0" class="data row17 col0" >x18</td>
      <td id="T_20da2_row17_col1" class="data row17 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_20da2_row18_col0" class="data row18 col0" >x19</td>
      <td id="T_20da2_row18_col1" class="data row18 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_20da2_row19_col0" class="data row19 col0" >x20</td>
      <td id="T_20da2_row19_col1" class="data row19 col1" >0.006608</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_20da2_row20_col0" class="data row20 col0" >x21</td>
      <td id="T_20da2_row20_col1" class="data row20 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_20da2_row21_col0" class="data row21 col0" >x22</td>
      <td id="T_20da2_row21_col1" class="data row21 col1" >0.006854</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_20da2_row22_col0" class="data row22 col0" >x23</td>
      <td id="T_20da2_row22_col1" class="data row22 col1" >0.119627</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_20da2_row23_col0" class="data row23 col0" >x24</td>
      <td id="T_20da2_row23_col1" class="data row23 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_20da2_row24_col0" class="data row24 col0" >x25</td>
      <td id="T_20da2_row24_col1" class="data row24 col1" >0.000045</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_20da2_row25_col0" class="data row25 col0" >x26</td>
      <td id="T_20da2_row25_col1" class="data row25 col1" >0.000404</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_20da2_row26_col0" class="data row26 col0" >x27</td>
      <td id="T_20da2_row26_col1" class="data row26 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_20da2_row27_col0" class="data row27 col0" >x28</td>
      <td id="T_20da2_row27_col1" class="data row27 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_20da2_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_20da2_row28_col0" class="data row28 col0" >x29</td>
      <td id="T_20da2_row28_col1" class="data row28 col1" >0.000000</td>
    </tr>
  </tbody>
</table>

Tree-based models such as Decision Trees, Random Forests, and Gradient Boosting Machines can provide feature importance scores. These scores represent the relative importance of each feature in the model's decision-making process. Features with higher importance scores are considered more significant in predicting the target variable. Let us try DecisionTreeClassifier from scikit-learn to train a decision tree model and then extract feature importances from it. 

```python
from sklearn.tree import DecisionTreeClassifier
# Initialize DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()

# Fit the model
dt_classifier.fit(X_scaled, y)

# Get feature importances
feature_importances = dt_classifier.feature_importances_
feature_importances
``` 

![image](https://github.com/LeThiKhanhHien/Notebooks/assets/56401620/dd66b839-b6f5-41a5-a40e-23bb1365f4fb)

## Explore some AD models for fraud credit card detection task 

We will try the following AD models 
- The usual logistic regression model 
- Decision Tree Classifier
- Isolation Forest
- Empirical Cumulative Distribution-based Outlier Detection

```python
scaler = StandardScaler()

logistic_regression = LogisticRegression()
dt_classifier = DecisionTreeClassifier()
isolation_forest = IsolationForest()
ecod = ECOD()

# Create pipelines with StandardScaler and models
pipeline_lr = Pipeline([('scaler', scaler), ('lr', logistic_regression)])
pipeline_dt = Pipeline([('scaler', scaler), ('dt', dt_classifier)])
pipeline_if = Pipeline([('scaler', scaler), ('if', isolation_forest)])
pipeline_ecod = Pipeline([('scaler', scaler), ('ecod', ecod)])

# Define multiple metrics
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'f1-score': make_scorer(f1_score, average='macro'),
           'auc_roc':make_scorer(roc_auc_score),
           'auc_pr': make_scorer(average_precision_score)
          }
```

## Evaluation ## 

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic regression</th>
      <th>decision tree</th>
      <th>isolated forest</th>
      <th>ECOD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.507102</td>
      <td>15.311816</td>
      <td>0.357572</td>
      <td>2.354623</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.056995</td>
      <td>0.051611</td>
      <td>0.239926</td>
      <td>2.824223</td>
    </tr>
    <tr>
      <th>test_accuracy</th>
      <td>0.999171</td>
      <td>0.999185</td>
      <td>0.037166</td>
      <td>0.901347</td>
    </tr>
    <tr>
      <th>test_precision</th>
      <td>0.936744</td>
      <td>0.879254</td>
      <td>0.481258</td>
      <td>0.507547</td>
    </tr>
    <tr>
      <th>test_recall</th>
      <td>0.804767</td>
      <td>0.890021</td>
      <td>0.100744</td>
      <td>0.893781</td>
    </tr>
    <tr>
      <th>test_f1-score</th>
      <td>0.858345</td>
      <td>0.884069</td>
      <td>0.035869</td>
      <td>0.489068</td>
    </tr>
    <tr>
      <th>test_auc_roc</th>
      <td>0.804767</td>
      <td>0.890021</td>
      <td>0.100744</td>
      <td>0.893781</td>
    </tr>
    <tr>
      <th>test_auc_pr</th>
      <td>0.533359</td>
      <td>0.592237</td>
      <td>0.001494</td>
      <td>0.013779</td>
    </tr>
  </tbody>
</table>
</div>

![image](https://github.com/LeThiKhanhHien/Notebooks/assets/56401620/74860edd-1c28-4986-a287-edfa600fcffc)

## Deployment ## 

We observe that decision tree has the slowest fitting time among the models but it has the best accuracy, recall, f1-score, auc-roc and auc-pr scores. Logistic regression is the second best in these scores, it has the best precision score, and much faster fitting time than the decision tree model. It seems logistic regression is a good supervised method for fraud card detection task. ECOD also works quite well. Note that ECOD is unsupervised method. 


