
# Here feature elimination technique is used to remove features that have very low correlation with label(price).

# Importing Libraries.

import pandas as pd
import numpy as np

# Creating dataframes.

train_df=pd.DataFrame(pd.read_excel('Cleaned_train.xlsx'))
test_df=pd.DataFrame(pd.read_excel('Cleaned_test.xlsx'))

train_feat=train_df.iloc[:,:-1]
train_label=train_df.iloc[:,-1]

test_feat=test_df.iloc[:,:-1]
test_label=test_df.iloc[:,-1].values

# Feature elimination (Features which have very low correlation with label)

train_feat.drop('Inches',axis=1,inplace=True)
train_feat.drop('Type_Name',axis=1,inplace=True)

test_feat.drop('Inches',axis=1,inplace=True)
test_feat.drop('Type_Name',axis=1,inplace=True)


# Model training & testing

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=13,random_state=0)

model.fit(train_feat,train_label)
predicted_label=model.predict(test_feat)
predicted_label=predicted_label.round(3)

print(np.concatenate((test_label.reshape(-1,1),predicted_label.reshape(-1,1)),1))

# Evaluating model performance
from sklearn.metrics import mean_absolute_error ,mean_absolute_percentage_error,r2_score

r2=r2_score(test_label,predicted_label)
MAE=mean_absolute_error(test_label,predicted_label)
MAPE=mean_absolute_percentage_error(test_label,predicted_label)

n=1199  # Number of observations in train data set.
k=8    # Number of Independent variables(features)
adj_r2_score = 1 - ((1-r2)*(n-1)/(n-k-1))

print(f'mean absolute error is:',MAE)
print(f'mean absolute percentage error is:',MAPE)
print(f'R-Squares score is:',r2)
print(f'Adjusted R-Squared score is:',adj_r2_score)

'''
Result:
mean absolute error is: 159.38968888888888
mean absolute percentage error is: 0.16436154712327572
R-Squares score is: 0.8636026553318473
Adjusted R-Squared score is: 0.8626856983929018

'''
 