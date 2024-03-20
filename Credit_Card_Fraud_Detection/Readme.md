# Credit Card Fraud Detection # 

Author: Hien Le

<img src="https://securityintelligence.com/wp-content/webp-express/webp-images/doc-root/wp-content/uploads/2016/09/credit-card-fraud-remains-a-risk.jpg.webp" alt="Image Description" style="width: 300px;"/>

## Link to Blog ## 
[Link to blog post](https://medium.com/@khanhhiennt/credit-card-fraud-detection-7bebee9b9adc)

## Project motivation ## 
Credit card companies must possess the capability to detect fraudulent credit card transactions to ensure customers aren't billed for unauthorized purchases. We will explore some machine learning models to detect fraudulent credit card transactions.

Three questions we answer in the notebook
1. How can we compare the performance of different models?
2. Can we say somethings about the importance level of the features in the dataset? 
3. Explore some models and which one among them is the best for credit card fraud detection task? 

## Library used ##

```python
import pandas as pd
import matplotlib.pyplot  as plt
from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from pyod.models.ecod import ECOD
``` 

## Acknowledgement ##

The dataset is downloaded from [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) 
