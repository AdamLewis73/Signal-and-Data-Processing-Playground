import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import scipy.signal as signal
from scipy import fftpack
from matplotlib import pyplot as plt
from scipy.fftpack import fft
import pywt

from sklearn.cluster import KMeans

#file = 'C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_STATIC_TR01.csv'
#df23 = pd.read_csv(file, header = 5)[7:]
#print(df23)
#df.columns=['Index', 'Values']
#print(df)

#df1 = df.iloc[:len(df),:len(df.columns)]
#print(df1)

#df2 = df1.dropna(axis = 0, how = 'any')
#print(df2)
#print(len(df))
#print(len(df.columns))



##########################3

file = 'C:/Users/sword/Anaconda3/envs/exceltest/emgtest.csv'
df = pd.read_csv(file,nrows=5383)
emg1 = df.emg1
emg2 = df.emg2
print(len(df.emg1))
x = np.arange(0,len(df.emg1)/1000,1/1000)
#coef, freqs=pywt.dwt(emg1,'db1')
#print(coef)
#plt.matshow(coef)
#plt.show()
print(type(emg1))
print(type(x))
x = pd.DataFrame(x)
print(type(x))
#emg1re = emg1.reshape(-1,1)
conc = pd.concat([x,emg1],axis=1)
print(conc)
conc2 = pd.concat([x,emg2],axis=1)
ait = KMeans(n_clusters = 3).fit(conc)
pred = ait.predict(conc2)
print(pred)
xtemp = conc.values[:,0]
ytemp = conc.values[:,1]
plt.scatter(xtemp,ytemp,c = pred, alpha = 0.6)
# num_clust = list(range(1,9))
# inertias = []
# for i in num_clust:
#     ait = KMeans(n_clusters = i).fit(conc)
#     inertia = ait.inertia_
#     inertias.append(inertia)

# plt.subplot(1,2,1)
# plt.plot(x,emg2)
# plt.subplot(1,2,2)
# plt.plot(num_clust,inertias)
#
#
plt.show()