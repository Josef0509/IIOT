# %%
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database import DB_c

myDB = DB_c.DB()

# Fetch data from database and create DataFrames
final_weight = pd.DataFrame(myDB.query("SELECT * FROM finalWeight"), columns=['bottle', 'time', 'final_weight'])
disp_vibration = pd.DataFrame(myDB.query("SELECT * FROM DispVibration"), columns=['color', 'bottle', 'time', 'vibration_index'])
dispenser = pd.DataFrame(myDB.query("SELECT * FROM Dispenser"), columns=['color', 'bottle', 'time', 'fill_level_grams', 'recipe'])
temperature = pd.DataFrame(myDB.query("SELECT * FROM Temperature"), columns=['time', 'temperature_c'])
drop_vibration = pd.DataFrame(myDB.query("SELECT * FROM dropVibration"), columns=['bottle', 'n', 'dropVibration'])
ground_truth = pd.DataFrame(myDB.query("SELECT * FROM ground_truth"), columns=['bottle', 'is_cracked'])


disp_vibration_red = disp_vibration[disp_vibration['color'] == 'red']
disp_vibration_red = disp_vibration_red.drop('color', axis=1)
disp_vibration_red = disp_vibration_red.drop('bottle', axis=1)
disp_vibration_red = disp_vibration_red.rename(columns={'vibration_index': 'vibration-index_red_vibration'})

disp_vibration_green = disp_vibration[disp_vibration['color'] == 'green']
disp_vibration_green = disp_vibration_green.drop('color', axis=1)
disp_vibration_green = disp_vibration_green.drop('bottle', axis=1)
disp_vibration_green = disp_vibration_green.rename(columns={'vibration_index': 'vibration-index_green_vibration'})

disp_vibration_blue = disp_vibration[disp_vibration['color'] == 'blue']
disp_vibration_blue = disp_vibration_blue.drop('color', axis=1)
disp_vibration_blue = disp_vibration_blue.drop('bottle', axis=1)
disp_vibration_blue = disp_vibration_blue.rename(columns={'vibration_index': 'vibration-index_blue_vibration'})


dispenser_red = dispenser[dispenser['color'] == 'red']
dispenser_red = dispenser_red.drop('color', axis=1)
dispenser_red = dispenser_red.drop('bottle', axis=1)
dispenser_red = dispenser_red.drop('recipe', axis=1)
dispenser_red = dispenser_red.rename(columns={'fill_level_grams': 'fill_level_grams_red'})  

dispenser_green = dispenser[dispenser['color'] == 'green']
dispenser_green = dispenser_green.drop('color', axis=1)
dispenser_green = dispenser_green.drop('bottle', axis=1)
dispenser_green = dispenser_green.drop('recipe', axis=1)
dispenser_green = dispenser_green.rename(columns={'fill_level_grams': 'fill_level_grams_green'})

dispenser_blue = dispenser[dispenser['color'] == 'blue']
dispenser_blue = dispenser_blue.drop('color', axis=1)
dispenser_blue = dispenser_blue.drop('bottle', axis=1)
dispenser_blue = dispenser_blue.drop('recipe', axis=1)
dispenser_blue = dispenser_blue.rename(columns={'fill_level_grams': 'fill_level_grams_blue'})


temperature_mean = temperature.groupby('time')['temperature_c'].mean().reset_index()
temperature_mean = temperature_mean.rename(columns={'temperature_c': 'temperature_mean_C'})
# drop first column

# subtract 2 from time from final_weight
final_weight['time'] = final_weight['time'] - 2


df = temperature_mean
df = pd.merge(df, disp_vibration_red, on=['time'], how='inner')
df = pd.merge(df, disp_vibration_green, on=['time'], how='inner')
df = pd.merge(df, disp_vibration_blue, on=['time'], how='inner')

df = pd.merge(df, dispenser_red, on=['time'], how='inner')
df = pd.merge(df, dispenser_green, on=['time'], how='inner')
df = pd.merge(df, dispenser_blue, on=['time'], how='inner')

df = pd.merge(df, final_weight, on=['time'], how='inner')


df = df.drop('time', axis=1)
df = df.drop('bottle', axis=1)

y = final_weight['final_weight']
df = df.drop('final_weight', axis=1)

x = df


print(x.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)
X_train.head()


import numpy as np
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

model.coef_
# %%
# Prediction Training

y_pred = model.predict(X_train)

# make MSE for y_pred of training data
from sklearn.metrics import mean_squared_error
mse_training = mean_squared_error(y_train, y_pred)
mse_training
# %%
# Prediction Testing
y_pred = model.predict(X_test)

# make MSE for y_pred of training data
from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test, y_pred)
mse_test
# %%

# make table of mse; first column temperature_mean_C, disp_vibration_red, disp_vibration_green, disp_vibration_blue, fill_level_grams_red, fill_level_grams_green, fill_level_grams_blue, second column linear model type, third column mse-training, fourth column mse-test
mse_table = pd.DataFrame(columns=['Genutzte Spalten', 'linear model type', 'mse-training', 'mse-test'])
#fill genutzet spalten
mse_table['Genutzte Spalten'] = ['temperature_mean_C, disp_vibration_red, disp_vibration_green, disp_vibration_blue, fill_level_grams_red, fill_level_grams_green, fill_level_grams_blue']
#fill linear model type
mse_table['linear model type'] = 'Linear Regression'
#fill mse-training
mse_table['mse-training'] = mse_training
#fill mse-test
mse_table['mse-test'] = mse_test
mse_table

#table to csv
mse_table.to_csv('mse_table.csv')

# %%

#import csv to pd.dataframe
df = pd.read_csv('X.csv')
#drop first column
df = df.drop(df.columns[0], axis=1)
df

# %%
#predict
y_pred = model.predict(df)

df = pd.DataFrame({'Predicted': y_pred})

#export the predictions to csv
df.to_csv('reg_52216061-62100348.csv')

#plot the predictions
plt.plot(y_pred)
plt.show()
