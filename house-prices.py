import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from sklearn.metrics import confusion_matrix
from tabulate import tabulate

def process_data(data):

    data = data.drop(columns=['Id'])

    
    data['LotFrontage'].fillna(0,inplace=True)
    data['MasVnrArea'].fillna(0,inplace=True)
    data['GarageYrBlt'].fillna(0,inplace=True)
    data['BsmtFinSF1'].fillna(0,inplace=True)
    data['BsmtFinSF2'].fillna(0,inplace=True)
    data['BsmtUnfSF'].fillna(0,inplace=True)
    data['TotalBsmtSF'].fillna(0,inplace=True)
    data['BsmtFullBath'].fillna(0,inplace=True)
    data['BsmtHalfBath'].fillna(0,inplace=True)
    data['GarageCars'].fillna(0,inplace=True)
    data['GarageArea'].fillna(0,inplace=True)

    column_names = data.head(0)
    for column in column_names:
        if data[column].dtype == 'object':
            values = pd.unique(data[column])
            for i in range(values.shape[0]):
                data.replace({column:{values[i]:i}},inplace=True)

    return data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

#read in data
data = pd.read_csv('train.csv')

data = process_data(data)

label = data['SalePrice']
train = data.drop(columns=['SalePrice'])

x_train,x_test,y_train,y_test=train_test_split(train,label,test_size=0.25,random_state=2)

model = Ridge(0.1)
model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print("Accuracy = ",accuracy)

#test data

test = pd.read_csv('test.csv')

print(test.shape)

test = process_data(test)

predictions = model.predict(test)
print(test.shape,predictions.shape)

pd.DataFrame(predictions).to_csv('houseprice.csv')



    






