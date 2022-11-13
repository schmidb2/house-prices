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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

#read in data
data = pd.read_csv('train.csv')
data = data.drop(columns=['Id'])
na_values = data.isnull().sum()

data['LotFrontage'].fillna(0,inplace=True)
data['MasVnrArea'].fillna(0,inplace=True)
data['GarageYrBlt'].fillna(0,inplace=True)

#replace string features with ints
column_names = data.head(0)
for column in column_names:
    if data[column].dtype == 'object':
        values = pd.unique(data[column])
        for i in range(values.shape[0]):
            data.replace({column:{values[i]:i}},inplace=True)

label = data['SalePrice']
train = data.drop(columns=['SalePrice'])

x_train,x_test,y_train,y_test=train_test_split(train,label,test_size=0.25,random_state=2)

model = Ridge(0.1)
model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print("Accuracy = ",accuracy)





