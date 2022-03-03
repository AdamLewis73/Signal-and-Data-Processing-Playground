### CSV Extraction Code with help from: Jim Todd of Stack Overflow
from scipy import fftpack, signal
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
temp = []
samplerate = 1000
nyq = samplerate*0.5


# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_MVIC_PreGAS_TR01.csv', 'r') as csvfile:
with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_MVIC_PreBiF_TR01.csv', 'r') as csvfile:
#with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_MVIC_PreRF_TR01.csv', 'r') as csvfile:
#with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_MVIC_PreTA_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_STATIC_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_STATIC_TR02.csv', 'r') as csvfile:
     csvreader = csv.reader(csvfile, delimiter=',')
     for row in csvreader:
         if csvreader.line_num == 3:
             temp.append(row)
         if csvreader.line_num >= 6:
            if row:
                temp.append(row)
            else:
                break
df = pd.DataFrame(temp)
df.columns = df.iloc[0]
#print(df)

df = df.drop(0)
df.reindex(df.index.drop(1))
df.reset_index(drop = True, inplace = True)
#print(df.columns)
#print(len(df))
#print(df)
#print(df['Noraxon Desk Receiver - EMG1'])

emgtemp = df['Noraxon Desk Receiver - EMG1']
emg1 = emgtemp.astype(np.float)
#emg1 = emg2[2968:28140]
hor = np.arange(0, len(emg1)/samplerate, 1/samplerate)
print(len(emg1))
# lowcut = 10
# N=4
# Wn= (2*3.14*lowcut)/nyq
# # N, Wn = signal.buttord(, , 3, 16)
# B, A = signal.butter(N, Wn, 'highpass')
# tempdf = signal.filtfilt(B, A, emg1)

cutoff_freq = 20 # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
cut = cutoff_freq/nyq
b,a = signal.butter(5,cut, btype ='highpass', analog = False)
yfilt = signal.filtfilt(b,a,emg1)

cutoff_freq = 400 # ~500 Hz according to the emg book
cut = cutoff_freq/nyq
b,a = signal.butter(5,cut, btype ='lowpass', analog = False)
yfilt2 = signal.filtfilt(b,a,yfilt)
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

emgfft = fftpack.fft(yfilt2, hor.size)
emgfftabs = np.abs(emgfft)
xf = fftpack.fftfreq(hor.size, (len(yfilt2) / samplerate) / samplerate) #finding the length in seconds divided by samples

plt.figure()
plt.subplot(2,2,1)
plt.plot(hor, emg1)

plt.subplot(2,2,2)
plt.plot(xf[0:len(xf)//2],(2/hor.size)*emgfftabs[0:len(emgfftabs)//2])

plt.subplot(2,2,3)
f2, PXX_den = signal.periodogram(yfilt2, samplerate)
meantest2 = sum(f2*PXX_den)/sum(PXX_den)
print(meantest2)
plt.semilogy(f2,PXX_den)
plt.ylim([1e-17,1e-8])
plt.show()
