#%%
import sys
import os
import pandas as pd
from scipy.fft import fft
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database import DB_c

myDB = DB_c.DB()

dropVibr = pd.DataFrame(myDB.getDropVibr())
cracked = pd.DataFrame(myDB.getBottleCracked())

# add column names
dropVibr.columns = ['bottle', 'n', 'dropVibration']
cracked.columns = ['bottle', 'is_cracked']

#%%     Look at drop vibration data
dropVibr

#%%     Look at cracked bottle data
cracked

# %%    extract drop vibrations for each bottle and calculate fft
fft_data = []

for bottle in dropVibr['bottle'].unique():
    bottle_data = dropVibr[dropVibr['bottle'] == bottle]['dropVibration'].values
    fft_data.append(np.abs(fft(bottle_data)))

fft_data = pd.DataFrame(fft_data)

# %%    plot the fft data of a good and a cracked bottle
plt.plot(fft_data.iloc[0])
plt.title('Good bottle')
plt.show()

plt.plot(fft_data.iloc[20])
plt.title('Cracked bottle')
plt.show()

# %%    cut the fft in half and delete the right half, it is mirrored
fft_data = fft_data.iloc[:, :len(fft_data.columns) // 2]

plt.plot(fft_data.iloc[0])
plt.title('Good bottle, half fft')
plt.show()

# %%    find the max value for each bottle, and classify each value over a threshold as peak
max_values = fft_data.max(axis=1)
thresholds = np.array(0.3*max_values)

#%%     get the frequencies of the peaks
is_peak = fft_data > thresholds[:, None]
# get the frequencys of the peaks
f_peak = is_peak.apply(lambda x: x.index[x].tolist(), axis=1)
# only allow 3 peaks, discard the rest
f_peak = f_peak.apply(lambda x: x[:3])

# remove the last rows to have equal length
f_peak = f_peak[:-2]

# unpack the list uín the firat column to 3 columns
f_peak = pd.DataFrame(f_peak.to_list(), columns=['f1', 'f2', 'f3'])
data = f_peak.values
knn = KNeighborsClassifier(n_neighbors=5)
classes = cracked['is_cracked'].values

# %%    fit on complete data

knn.fit(data, classes)
predictions = knn.predict(data)
accuracy = np.mean(predictions == classes)
print("Accuracy with 100% trainset: "+str(accuracy))

# create confusion matrix
confusion_matrix = pd.crosstab(classes, predictions, rownames=['Actual'], colnames=['Predicted'])
confusion_matrix
#calculate f1 score
f1 = 2*confusion_matrix.iloc[1,1]/(2*confusion_matrix.iloc[1,1]+confusion_matrix.iloc[0,1]+confusion_matrix.iloc[1,0])
print("F1 score: "+str(f1))

# %%   fit on 80% of the data and test on the other 20%
split = int(len(data)*0.8)
knn.fit(data[:split], classes[:split])
predictions = knn.predict(data[split:])
accuracy = np.mean(predictions == classes[split:])
print("Accuracy with 80% trainset: "+str(accuracy))

# create confusion matrix
confusion_matrix = pd.crosstab(classes[split:], predictions, rownames=['Actual'], colnames=['Predicted'])
confusion_matrix
#calculate f1 score
f1_1 = 2*confusion_matrix.iloc[1,1]/(2*confusion_matrix.iloc[1,1]+confusion_matrix.iloc[0,1]+confusion_matrix.iloc[1,0])
print("F1 score: "+str(f1_1))

# %% 

klass_table = pd.DataFrame(columns=['Genutzte Features', 'Modell-Typ', 'F1-Score (Training)', 'F1-Score (Test)'])
klass_table.loc[0] = ['dropVibration (fft -> threshold of 0.3*max -> 3 Frequenzen der Peaks)', 'KNN mit 5 Nachbarn', f1, f1_1]

klass_table
#%%