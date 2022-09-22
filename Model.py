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

n=90   # Number of observations in test set.
k=2    # Number of Independent variables(features)
adj_r2_score = 1 - ((1-r2)*(n-1)/(n-k-1))


print(f'mean absolute error is:',MAE)
print(f'mean absolute percentage error is:',MAPE)
print(f'R-Squares score is:',r2)
print(f'Adjusted R-Squared score is:',adj_r2_score)


'''
Result:
mean absolute error is: 136.38088888888888
mean absolute percentage error is: 0.1445928277205149
R-Squares score is: 0.8899338480010867
Adjusted R-Squared score is: 0.8874035916332955
'''