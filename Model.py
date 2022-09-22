# Importing Libraries.

import pandas as pd
import numpy as np

# Creating dataframes.

train_df=pd.DataFrame(pd.read_excel('Cleaned_train.xlsx'))
test_df=pd.DataFrame(pd.read_excel('Cleaned_test.xlsx'))

train_feat=train_df.iloc[:,:-1]
train_label=train_df.iloc[:,-1].values

train_label=train_label.reshape(-1,1)

test_feat=test_df.iloc[:,:-1]
test_label=test_df.iloc[:,-1].values

# Feature Scaling

from sklearn.preprocessing import StandardScaler
SC_feat=StandardScaler()
SC_label=StandardScaler()

train_feat=SC_feat.fit_transform(train_feat)
test_feat=SC_feat.transform(test_feat)
train_label=SC_label.fit_transform(train_label)


# Model training & testing

from sklearn.svm import SVR
model=SVR(kernel='rbf')

model.fit(train_feat,train_label)

x=model.predict(test_feat)
x=x.reshape(-1,1)
predicted_label=SC_label.inverse_transform(x)
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
mean absolute error is: 193.19833333333335
mean absolute percentage error is: 0.20314361080660118
R-Squares score is: 0.7577458788129114
Adjusted R-Squared score is: 0.7521768185557369

'''