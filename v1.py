# -*- coding: utf-8 -*-
"""
Created on Nov 2019

@author: Sagar Paithankar
"""

import os
os.chdir(r'E:\My Drive\Enquero')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
#read excel
#sales is copy of test evaluation
raw = pd.read_excel('Sales.xlsx')
#columns
['item', 'Division', 'weekNumber', 'Flag_Occurence', 'month', 'quarter','half', 'Quantity']
#split the column into year and woy
raw['weekNumber'] = raw['weekNumber'].astype('str')
raw['year'] = (raw['weekNumber'].apply(lambda x : '20'+ x[:2])).astype('int')
raw['woy'] = (raw['weekNumber'].apply(lambda x : x[2:])).astype('int')
quarter = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}
raw['quarter'] = raw['month'].map(quarter)
#finding null values
raw.isna().sum()
#finding unique values
uni = {}
for col in raw.columns:
    uni.update({col : list(raw[col].unique())})
#droping some columns
raw.drop(columns=['Division','weekNumber'], inplace=True)
#finding unique values
uni = {}
for col in raw.columns:
    uni.update({col : list(raw[col].unique())})

['item', 'Flag_Occurence', 'month', 'quarter', 'half', 'Quantity','year', 'woy']
#plot statistics values
desc = raw.describe()
'''
>>this will show mean , max, min std and hence outlier can be found by this
>>try to boxplot
>>find quantile wise
>>find z score for entire data set
'''
sns.boxplot(raw['Quantity'])
out = raw[raw['Quantity'] > raw['Quantity'].quantile(0.99)]
z = np.abs(stats.zscore(raw['Quantity']))
outl = raw[z > 3]

#finding Interquartile range IQR
Q1 = raw['Quantity'].quantile(0.25)
Q3 = raw['Quantity'].quantile(0.75)
IQR = Q3 - Q1

#reading final prediction excel
result= {}
to_pred = pd.read_excel('forecst.xlsx')

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>case 1 >>>>>>>>Removing outlire 

df = raw[(raw['Quantity'] > Q1 - 1.5*IQR) & (raw['Quantity'] < Q3 + 1.5*IQR)]
#finding unique values
uni = {}
for col in df.columns:
    uni.update({col : list(df[col].unique())})

['item', 'Flag_Occurence', 'month', 'quarter', 'half', 'Quantity','year', 'woy']
#plot statistics values
desc = df.describe()
df['Quantity'].plot()
sns.boxplot(df['Quantity'])
#finding correlation
corr = df.corr()
corr.plot()


#>>>>>>statsmodels
#scaling the variable
from sklearn.preprocessing import StandardScaler
sr = StandardScaler()
x = sr.fit_transform(df.loc[:, ['item','Flag_Occurence','month','quarter','half','year', 'woy']])
y = df.loc[:, 'Quantity']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#['item','Flag_Occurence','month','quarter','half','year', 'woy']
X = sm.add_constant(x_train) # adding a constant
model = sm.OLS(y_train, X).fit() 
print(model.summary())


#removed 'year'
X = X[:,[0,1,2,3,4,5,7]]
#['item','Flag_Occurence','month','quarter','half', 'woy']
model = sm.OLS(y_train, X).fit() 
print(model.summary())

#removed 'quarter'
X = X[:,[0,1,2,3,5,6]]
#['item','Flag_Occurence','month','half', 'woy']
model = sm.OLS(y_train, X).fit() 
print(model.summary())

#removed 'Flag_Occurence'
#['item','month','half', 'woy']
X = X[:,[0,1,3,4,5]]
model = sm.OLS(y_train, X).fit() 
print(model.summary())

#removed 'month'
#['item','half', 'woy']
X = X[:,[0,1,3,4]]
model = sm.OLS(y_train, X).fit() 
print(model.summary())


X_test = sm.add_constant(x_test)
#['item','Flag_Occurence','month','quarter','half','year', 'woy']
#[0,1,5,7]
y_pred = model.predict(X_test[:,[0,1,5,7]])
a = mean_absolute_error(y_test, y_pred)
a*100/ np.mean(y_test)

x_pred = to_pred.loc[:, ['item', 'Flag_Occurence', 'month', 'quarter', 'half', 'year', 'woy']]
y = sr.transform(x_pred)
y = np.append(arr = np.ones((15,1)).astype(int), values = y, axis = 1)
y = y[:, [0,1,4,6]]
forc = model.predict(y)
result.update({'OLS':forc})

#>>>>>>>>SVR

#scaling the variable
from sklearn.preprocessing import StandardScaler
sr = StandardScaler()
x = sr.fit_transform(df.loc[:, ['item','half','woy']])
y = df.loc[:, 'Quantity']

from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


parameters = {'epsilon':[0.1, 0.05, 0.01, 1, 65, 50, 100, 75, 95],\
              'C':[0.01, 0.1, 1, 100, 500, 1000, 3000, 4000],\
              'degree':[3, 5, 7]}
from sklearn.svm import SVR
model = SVR(gamma='auto')
clf = GridSearchCV(model, parameters, cv=5)
clf.fit(x_train, y_train)
clf.best_estimator_


model = SVR(C=500, cache_size=200, coef0=0.0, degree=3, epsilon=50,\
            gamma='auto', kernel='rbf', max_iter=-1, shrinking=True,\
            tol=0.001, verbose=False)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

a = mean_absolute_error(y_test, y_pred)
a*100/ np.mean(y_test)

x_pred = to_pred.loc[:, ['item', 'half', 'woy']]
y = sr.transform(x_pred)
forc = model.predict(y)
result.update({'SVC':forc})

#>>>>>>>>>LR
#scaling the variable
from sklearn.preprocessing import StandardScaler
sr = StandardScaler()
x = sr.fit_transform(df.loc[:, ['item','half', 'woy']])
y = df.loc[:, 'Quantity']

from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
y_pred = regr.predict(x_test)
a = mean_absolute_error(y_test, y_pred)
a*100/ np.mean(y_test)

x_pred = to_pred.loc[:, ['item', 'half', 'woy']]
y = sr.transform(x_pred)
forc = model.predict(y)
result.update({'LR':forc})


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##>>>>>>>>>>>>>case 2 >>>>>>>>replace outliers with np.Nan

df = raw.copy()
df.loc[(df['Quantity'] < Q1 - 1.5*IQR) | (df['Quantity'] > Q3 + 1.5*IQR), 'Quantity'] = np.nan
df.isna().sum()

df = df.fillna(df.mean())
df.isna().sum()

#finding unique values
uni = {}
for col in df.columns:
    uni.update({col : list(df[col].unique())})

['item', 'Flag_Occurence', 'month', 'quarter', 'half', 'Quantity','year', 'woy']
#plot statistics values
desc = df.describe()
df['Quantity'].plot()
sns.boxplot(df['Quantity'])
#finding correlation

#>>>>>>>

#scaling the variable
from sklearn.preprocessing import StandardScaler
sr = StandardScaler()
x = sr.fit_transform(df.loc[:, ['item','Flag_Occurence','month','quarter','half','year', 'woy']])
y = df.loc[:, 'Quantity']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#['item','Flag_Occurence','month','quarter','half','year', 'woy']
# with statsmodels
X = sm.add_constant(x_train) # adding a constant
model = sm.OLS(y_train, X).fit() 
print(model.summary())


#removed 'year'
X = X[:,[0,1,2,3,4,5,7]]
#['item','Flag_Occurence','month','quarter','half', 'woy']
model = sm.OLS(y_train, X).fit() 
print(model.summary())

#removed 'quarter'
X = X[:,[0,1,2,3,5,6]]
#['item','Flag_Occurence','month','half', 'woy']
model = sm.OLS(y_train, X).fit() 
print(model.summary())

#removed 'half'
#['item','Flag_Occurence','month', 'woy']
X = X[:,[0,1,2,3,5]]
model = sm.OLS(y_train, X).fit() 
print(model.summary())

#removed 'Flag_Occurence'
#['item','month', 'woy']
X = X[:,[0,1,3,4]]
model = sm.OLS(y_train, X).fit() 
print(model.summary())


X_test = sm.add_constant(x_test)
#['item','Flag_Occurence','month','quarter','half','year', 'woy']
#[0,1,3,6]
y_pred = model.predict(X_test[:,[0,1,3,7]])
a = mean_absolute_error(y_test, y_pred)
a*100/ np.mean(y_test)

x_pred = to_pred.loc[:, ['item','Flag_Occurence','month','quarter','half','year', 'woy']]
x_pred = sr.transform(x_pred)
x_pred = np.append(arr = np.ones((15,1)).astype(int), values = x_pred, axis = 1)
y = x_pred[:,[0,1,3,7]]
forc = model.predict(y)
result.update({'OLS1':forc})

#>>>>>>>

#scaling the variable
from sklearn.preprocessing import StandardScaler
sr = StandardScaler()
x = sr.fit_transform(df.loc[:, ['item','month', 'woy']])
y = df.loc[:, 'Quantity']

from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

parameters = {'epsilon':[0.1, 0.05, 0.01, 1, 65, 50, 100, 75, 95],\
              'C':[0.01, 0.1, 1, 100, 500, 1000, 3000, 4000],\
              'degree':[3, 5, 7]}
from sklearn.svm import SVR
model = SVR(gamma='auto', verbose=False)
clf = GridSearchCV(model, parameters, cv=5)
clf.fit(x_train, y_train)
clf.best_estimator_


model = SVR(C=3000, cache_size=200, coef0=0.0, degree=3, epsilon=50,\
            gamma='auto', kernel='rbf', max_iter=-1, shrinking=True,\
            tol=0.001, verbose=False)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

a = mean_absolute_error(y_test, y_pred)
a*100/ np.mean(y_test)


forecst = to_pred.loc[:, ['item','month', 'woy']]
y = sr.transform(forecst)
forc = model.predict(y)
result.update({'SVC1':forc})

#>>>>>>>>>LR
#scaling the variable
from sklearn.preprocessing import StandardScaler
sr = StandardScaler()
x = sr.fit_transform(df.loc[:, ['item','month', 'woy']])
y = df.loc[:, 'Quantity']

from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
y_pred = regr.predict(x_test)
a = mean_absolute_error(y_test, y_pred)
a*100/ np.mean(y_test)

x_pred = to_pred.loc[:, ['item', 'half', 'woy']]
y = sr.transform(x_pred)
forc = model.predict(y)
result.update({'LR1':forc})


#>>>>>>>> all results in final
final = pd.DataFrame(result)
final['final' ] =  final.mean(axis=1)
to_pred['Quantity'] = final.mean(axis=1)

to_pred.to_excel('final.xlsx', index=False)

