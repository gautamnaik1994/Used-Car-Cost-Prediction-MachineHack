# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:26:45 2019

@author: Gautam
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns

df_train=pd.read_excel('./dataset/Data_Train.xlsx')
#df_test=pd.read_excel('./dataset/Data_Test.xlsx')

to_drop = ['Name','New_Price','Location','Transmission','Owner_Type']

sns.pairplot(df_train);
sns.distplot(df_train['Price'])
sns.heatmap(df_train.corr(),annot=True)

def clean_data(data):
    data = data.drop(to_drop,axis=1)
    data['Mileage']= pd.to_numeric(data['Mileage'].str.extract(r'^(\d*)', expand=False)) 
    data['Engine']=pd.to_numeric(data['Engine'].str.extract(r'^(\d*)', expand=False))
    data['Power']=pd.to_numeric(data['Power'].str.extract(r'^(\d*)', expand=False))
    data.fillna(data.mean(),inplace=True)
    return data

def hot_encode(data):
    labelencoder = LabelEncoder()
    data[:, 2] = labelencoder.fit_transform(data[:, 2])
    onehotencoder = OneHotEncoder(categorical_features = [2])
    data = onehotencoder.fit_transform(data).toarray()
    return data
    
df_train = clean_data(data=df_train)
X= df_train.iloc[:, :-1].values
#X=df_train.drop('Price',axis=1)
X=hot_encode(data=X)
y = df_train['Price']
#df_test = clean_data(data=df_test)
#df_train = hot_encode(data=df_train)


#X_train = df_train.iloc[:, :-1].values
#y_train = df_train.iloc[:, 6].values
#X_train = X_train[:, 1:]

#X_test = df_test.iloc[:,:].values
#y_test = df_test.iloc[:, 6].values
#X_test = X_test[:, 1:]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)




from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
accuracy = regressor.score(X_test, y_test)





