import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import scipy.signal as signal
from scipy import fftpack
from matplotlib import pyplot as plt
from scipy.fftpack import fft
samplerate = 1000

file = 'C:/Users/sword/Anaconda3/envs/exceltest/emgtest.csv'
df = pd.read_csv(file,nrows=5383)
emg1 = df.emg1
print(len(df.emg1))
# hor = list(0:1:len(df.emg1))
hor = np.arange(0,len(df.emg1)/samplerate,1/samplerate)
print(hor)
nyq = samplerate*0.5
# lowcut = 10
# N=4
# Wn= (2*3.14*lowcut)/nyq
# # N, Wn = signal.buttord(, , 3, 16)
# B, A = signal.butter(N, Wn, 'highpass')
# tempdf = signal.filtfilt(B, A, emg1)
# emgfft = fftpack.fft(tempdf, hor.size)
# emgfftabs = np.abs(emgfft)
# xf = fftpack.fftfreq(hor.size, (len(df.emg1)/samplerate)/samplerate)
#
# meandf = 2*emgfftabs[0:len(emgfftabs)//2]*xf[0:len(xf)//2]
# mean1 = np.mean(meandf)
# meanamp = np.mean(2*emgfftabs[0:len(emgfftabs)//2])
# print(mean1)
# print(meanamp)
# print(mean1/meanamp)

emgfft = fftpack.fft(emg1, hor.size)
emgfftabs = np.abs(emgfft)
xf = fftpack.fftfreq(hor.size, (len(df.emg1) / samplerate) / samplerate)

meandf = 2*emgfftabs[0:len(emgfftabs)//2]*xf[0:len(xf)//2]
mean1 = np.mean(meandf)
meanamp = np.mean(2*emgfftabs[0:len(emgfftabs)//2])
print(mean1)
print(meanamp)
print(mean1/meanamp)



plt.figure()
plt.subplot(1,2,1)
plt.plot(hor, emg1)

plt.subplot(1,2,2)
plt.plot(xf[0:len(xf)//2],2*emgfftabs[0:len(emgfftabs)//2])
plt.show()
