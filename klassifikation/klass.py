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

# remove the last row
f_peak = f_peak[:-2]

# unpack the list uín the firat column to 3 columns
f_peak = pd.DataFrame(f_peak.to_list(), columns=['f1', 'f2', 'f3'])


# %%
knn = KNeighborsClassifier(n_neighbors=3)
data = f_peak.values
classes = cracked['is_cracked'].values

data[1]
knn.fit(data, classes)



#%%
new_f1 = 15
new_f2 = 60
new_f3 = 140
new_point = np.array([new_f1, new_f2, new_f3],)
new_point = new_point.reshape(1, -1)
print(new_point)

prediction = knn.predict(new_point)
# %%
prediction
# %%
