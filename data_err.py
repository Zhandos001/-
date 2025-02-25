import pandas as pd 
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt 

df = pd.read_csv("C:\\Users\\zhand\\Desktop\\хрень с рабочего стола\\data.csv")


df.columns = df.columns.str.lower().str.replace(' ','_')
df.engine_fuel_type = df.engine_fuel_type.str.replace('(required)','')
df.engine_fuel_type = df.engine_fuel_type.str.replace('(recommended)','')
df.drop_duplicates(inplace=True)
df['engine_fuel_type'] = df['engine_fuel_type'].fillna(df['engine_fuel_type'].mode()[0])
log_price = np.log1p(df.msrp)
electric_with_hp = df[(df['engine_fuel_type'] == 'electric') & (df['engine_hp'] != 0.0) & (-df['engine_hp'].isna())]
df['engine_hp'] = df['engine_hp'].fillna(electric_with_hp['engine_hp'].mean())
df['engine_cylinders'] = df['engine_cylinders'].fillna(0)
df['number_of_doors'] = df['number_of_doors'].fillna(df['number_of_doors'].median())
df.drop('market_category' , axis=1 ,inplace=True)
df.reset_index(drop=True , inplace=True)



from sklearn.model_selection import train_test_split
X_train, X_test_and_valid, y_train, y_test_and_valid = train_test_split(df[df.columns[:-1]], df.msrp,test_size=0.4, random_state=42 ) 
X_test, X_valid, y_test, y_valid = train_test_split(X_test_and_valid, y_test_and_valid, test_size=0.5 ,random_state=42)

y_train_orig = y_train
y_test_orig = y_test
y_valid_orig = y_valid

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
y_valid = np.log1p(y_valid)


xi = [160.0 , 19 , 1385]
w0 = 7.01 
w = [0.04 , -0.08 , 0.002]
    

from sklearn.linear_model import LinearRegression
from sklearn import metrics

feature_coloumns = ['year', 'engine_hp' , 'engine_cylinders' , 'city_mpg' , 'popularity']
X_train , X_test =  X_train[feature_coloumns] , X_test[feature_coloumns]

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)  

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE: " (metrics.mean_absolute_error(pred, y_test)))
print("MSE: ", (metrics.mean_squared_error(pred, y_test)))
print("R2 score: ", (metrics.r2_score(pred, y_test)))


print(pred)
