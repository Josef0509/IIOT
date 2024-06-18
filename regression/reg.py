# %%
import sys
import json
import os
import pandas as pd
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database import DB_c

myDB = DB_c.DB()

#df = myDB.query("SELECT * FROM finalWeight")
# print(df)
# print(type(df))

final_weight = pd.DataFrame(myDB.query("SELECT * FROM finalWeight"), columns=['bottle', 'time', 'final_weight'])
disp_vibration = pd.DataFrame(myDB.query("SELECT * FROM DispVibration"), columns=['color', 'bottle', 'time', 'vibration_index'])
dispenser = pd.DataFrame(myDB.query("SELECT * FROM Dispenser"), columns=['color', 'bottle', 'time', 'fill_level_grams', 'recipe'])
temperature = pd.DataFrame(myDB.query("SELECT * FROM Temperature"), columns=['time', 'temperature_c'])
drop_vibration = pd.DataFrame(myDB.query("SELECT * FROM dropVibration"), columns=['bottle', 'n', 'dropVibration'])
ground_truth = pd.DataFrame(myDB.query("SELECT * FROM ground_truth"), columns=['bottle', 'is_cracked'])

df = pd.merge(final_weight, disp_vibration, on=['bottle', 'time'], how='outer')
# %%
plot = sns.pairplot(df, hue="final_weight")
# plot.savefig("pairplot.png")


# y = df['petal_length']
# X = df.drop(['petal_length','species'], axis=1)

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train.head()

# import numpy as np
# from sklearn.linear_model import LinearRegression

# model = LinearRegression()

# model.fit(X_train, y_train)

# model.coef_

# y_pred = model.predict(X_train)

# y_pred

# df = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})
# df['sepal_length'] = X_train['sepal_length']
# df['sepal_width'] = X_train['sepal_width']
# df['petal_width'] = X_train['petal_width']


# from sklearn.metrics import mean_squared_error

# mean_squared_error(y_train, y_pred)

# df["Error"] = (df["Actual"] - df["Predicted"])
# df["Squared Error"] =df["Error"] **2
# df

# df["Squared Error"].mean()

# y_pred = model.predict(X_test)

# mean_squared_error(y_test, y_pred)


# # Hier wird z.B. sichbar, dass der Fehler nahezu normalverteilt ist
# # Das bedeutet, dass der Fehler zufällig ist und keine systematischen Fehler vorliegen
# sns.histplot(df["Error"], kde=True)


# %%
