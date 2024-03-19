# %% [markdown]
# # Credit Card Fraud Detection # 
# 
# Author: Hien Le
# 
# <img src="https://securityintelligence.com/wp-content/webp-express/webp-images/doc-root/wp-content/uploads/2016/09/credit-card-fraud-remains-a-risk.jpg.webp" alt="Image Description" style="width: 300px;"/>
# 
# ## Context ## 
# Credit card companies must possess the capability to detect fraudulent credit card transactions to ensure customers aren't billed for unauthorized purchases.
# 
# Three questions we will answer in this notebook
# 1. How can we compare the performance of different models?
# 2. Can we say somethings about the importance level of the features in the dataset? 
# 3. Explore some models and which one among them is the best for credit card fraud detection task? 
# 
# ## The dataset ##
# The dataset is downloaded from [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
# 
# 
# The dataset comprises credit card transactions made by European cardholders in September 2013. Within a span of two days, there were 492 fraudulent transactions out of a total of 284,807 transactions. This dataset is characterized by significant imbalance, with the positive class (frauds) representing only 0.172% of all transactions.
# 
# The dataset includes solely numerical input variables resulting from a PCA transformation. Due to confidentiality constraints, the original features or additional background information about the data are not provided. Principal component analysis (PCA) was applied to derive features V1 through V28; however, features 'Time' and 'Amount' were not subjected to PCA transformation. 'Time' denotes the elapsed seconds between each transaction and the first recorded transaction in the dataset, while 'Amount' represents the transaction amount. The 'Class' feature serves as the response variable, taking a value of 1 in cases of fraud and 0 otherwise.

# %% [markdown]
# ## Data preparation and observation ## 
# 1. The data has the shape (284807, 31); there are 29 feature columns.
# 2. There are no nan values in the data
# 3. The dataset is imbalanced

# %%
import pandas as pd
df = pd.read_csv('creditcard.csv')
df.head()

# %%
df.info()

# %%
df.columns

# %%
df['Class'].value_counts()

# %%
#the dataset is imbalanced
import matplotlib.pyplot  as plt
transaction_counts = df['Class'].value_counts()
plt.pie(transaction_counts, labels=transaction_counts.index, autopct='%1.1f%%')
plt.title('Percentage of Fraud Transaction ( 1: fraud, 0: normal)')
plt.axis('equal')
plt.show()

# %%
class_1_df = df[df['Class'] == 1]  # Data with class 1
class_0_df = df[df['Class'] == 0]  # Data with class 0
plt.scatter(class_0_df['Time'], class_0_df['Amount'], color='blue', label='Class 0',s=5)
plt.scatter(class_1_df['Time'], class_1_df['Amount'], color='red', label='Class 1',s=5)
plt.xlabel('Time')
plt.ylabel('Amount')
plt.title('Scatter Plot of Time vs Amount ')
plt.legend()

# %%
# data
y=df['Class']
X=df.drop(columns=["Time","Class"])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# ## Q1: How can we compare the performance of different models? ##
# 
# In essence, Credit Card Fraud Detection involves identifying anomalies within credit card transactions, making it an anomaly detection task. When evaluating the performance of anomaly detection models, we typically rely on the following criteria. Note that positive = anomaly, negative = normality. 
# 
# |          | Predicted Negative | Predicted Positive |
# |----------|--------------------|--------------------|
# | Actual Negative | TN               | FP               |
# | Actual Positive | FN               | TP               |
# 
# 
# - **Precision and Recall**: Precision represents the proportion of correctly identified anomalies among all instances labeled as anomalies by the model, while recall measures the proportion of true anomalies that are correctly identified by the model.
#   $$Precision= \frac{TP}{TP+FP}$$
#   $$Recall= \frac{TP}{TP + FN} $$ 
# 
# - **F1 Score**: The F1 score is the harmonic mean of precision and recall and provides a balanced measure of a model's performance.
# $$F1 Score=2 \times \frac{Precision×Recall}{Precision+Recall}$$
# 
# - **Area Under the ROC Curve (AUC-ROC)**: The ROC curve plots the true positive rate against the false positive rate at various threshold settings. AUC-ROC provides a single scalar value that represents the overall performance of the model across different threshold settings.
# 
# - **Area Under the Precision-Recall Curve (AUC-PR)**: Similar to AUC-ROC, AUC-PR summarizes the precision-recall curve's performance, particularly useful when dealing with imbalanced datasets.

# %% [markdown]
# ## Q2. Can we say somethings about the importance level of the features in the dataset?
# 
# In fact, the feature importance depends much on the model we use. 
# 
# Let us first use Ordinary Least Squares (OLS) regression from the statsmodels library to fit a linear regression model and obtain p-values for each feature coefficient. These p-values indicate the statistical significance of each feature in predicting the target variable. **In the context of linear regression**, if the p-value is less than a chosen significance level (commonly 0.05), it suggests that the corresponding feature has a statistically significant relationship with the target variable; a high p-value suggests that the corresponding feature may not have a statistically significant relationship with the target variable. Features with low p-values are likely to be important predictors, while features with high p-values may be less relevant and could potentially be removed from the model. **Note that using p-value is sometimes controversial.**

# %%
from statsmodels.regression.linear_model import OLS
model = OLS(y, X_scaled).fit()
pvalues = pd.DataFrame(model.pvalues)
pvalues.reset_index(inplace=True)
pvalues.rename(columns={0: "pvalue", "index": "feature"}, inplace=True)
pvalues.style.background_gradient(cmap='RdYlGn')

#We observe that feature V23 has a high p-value when using OLS regression model. 

# %% [markdown]
# Tree-based models such as Decision Trees, Random Forests, and Gradient Boosting Machines can provide feature importance scores. These scores represent the relative importance of each feature in the model's decision-making process. Features with higher importance scores are considered more significant in predicting the target variable. Let us try DecisionTreeClassifier from scikit-learn to train a decision tree model and then extract feature importances from it.  

# %%
from sklearn.tree import DecisionTreeClassifier
# Initialize DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()

# Fit the model
dt_classifier.fit(X_scaled, y)

# Get feature importances
feature_importances = dt_classifier.feature_importances_
feature_importances 


# %%
# Plot feature importances
plt.bar(range(len(feature_importances)), feature_importances, tick_label=X.columns)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances from Decision Tree Classifier')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# ## Explore some AD models for fraud credit card detection task 
# 
# We will try the following AD models 
# - The usual logistic regression model 
# - Decision Tree Classifier
# - Isolation Forest
# - Empirical Cumulative Distribution-based Outlier Detection 

# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from pyod.models.ecod import ECOD

from sklearn.metrics import make_scorer, accuracy_score

# %%
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

# %% [markdown]
# ## Evaluation ## 

# %%
# Perform 5-fold cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores_lr =cross_validate(pipeline_lr, X, y, cv=cv, scoring=scoring)
scores_lr=pd.DataFrame(scores_lr)
scores_lr_mean=scores_lr.mean()

# %%
scores_dt =cross_validate(pipeline_dt, X, y, cv=cv, scoring=scoring)
scores_dt=pd.DataFrame(scores_dt)
scores_dt_mean=scores_dt.mean()

# %%
y_is=y.values
y_is=y_is*2-1
scores_if =cross_validate(pipeline_if, X, y_is, cv=cv, scoring=scoring)
scores_if=pd.DataFrame(scores_if)
scores_if_mean=scores_if.mean()

# %%
#this is an unsupervised method
scores_ecod =cross_validate(pipeline_ecod, X, y, cv=cv, scoring=scoring)
scores_ecod=pd.DataFrame(scores_ecod)
scores_ecod_mean=scores_ecod.mean()

# %%
results=pd.concat([scores_lr_mean,scores_dt_mean,scores_if_mean,scores_ecod_mean],axis=1)
results.columns=['logistic regression','decision tree','isolated forest','ECOD']

# %%
results

# %%
for index, row in results.drop(['fit_time','score_time']).iterrows():
    plt.plot(row.index, row.values, label=f'Mean {index}')

# Add labels and legend
plt.xlabel('Model')
plt.ylabel('Value')
plt.title('Line plot of each evaluation factors')
plt.legend()
plt.show()

# %% [markdown]
# ## Deployment ## 
# 
# We observe that decision tree has the slowest fitting time among the models but it has the best accuracy, recall, f1-score, auc-roc and auc-pr scores. Logistic regression is the second best in these scores, it has the best precision score, and much faster fitting time than the decision tree model. It seems logistic regression is a good supervised method for fraud card detection task. ECOD also works quite well. Note that ECOD is unsupervised method. 


