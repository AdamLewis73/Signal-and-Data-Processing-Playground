### CSV Extraction Code with help from: Jim Todd of Stack Overflow
from scipy import fftpack, signal, integrate
from matplotlib import pyplot as plt
from scipy.signal import hilbert,stft
from sklearn import preprocessing
import numpy as np
import pandas as pd
import csv
import tftb
import math
import scipy.signal as sps
temp = []
samplerate = 1500
nyq = samplerate*0.5


# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg_Fourth/Adam_Biodex_Leg4_TR02.csv', 'r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT01/Free/RF2_SubjP01_Free_Biodex_TR01_Fixed.csv', 'r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT01/Free/RF2_SubjP01_PostBiodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT02/Free/RF2_Subj02_Free_Biodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT02/Free/RF2_Subj02_Free_PostBiodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT03/Free/RF2_Subj03_Free_Biodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT03/Free/RF2_Subj03_Free_PostBiodex_TR01_Fixed.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT05/Free/RF2_Subj05_Free_Biodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT05/Free/RF2_Subj05_Free_PostBiodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT06/Free/RF2_Subj06_Free_Biodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT06/Free/RF2_Subj06_Free_PostBiodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT07/Free/RF2_Subj07_Free_PreBiodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT07/Free/RF2_Subj07_Free_PostBiodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT09/Free/RF2_Subj09_Free_Biodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT09/Free/RF2_Subj09_Free_PostBiodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT11/Free/RF2_Subj11_Free_PreBiodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT11/Free/RF2_Subj11_Free_PostBiodex_TR01_Fixed.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT12/Free/RF2_Subj12_Free_PreBiodex_TR01.csv','r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT12/Free/RF2_Subj12_Free_PostBiodex_TR01.csv','r') as csvfile:
with open('C:/Users/sword/PycharmProjects/excelimport/PT10/Free/RF2_Subj10_Free_Fatigue_TR36.csv','r') as csvfile:


    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        if csvreader.line_num == 3:
            temp.append(row)
        if csvreader.line_num >= 6:
            if row:
                temp.append(row)
            else:
                break

df = pd.DataFrame(temp)  # turns the array into a dataframe
df.columns = df.iloc[0]  # sets the column names as the first row
df = df.drop(0)  # drops the first row since it is now a duplicate of the column names
df.reindex(df.index.drop(1))
df.reset_index(drop=True, inplace=True)
#print(df)
df.columns = ['frames', 'subframes', 'blank', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8', 'EMG9', 'EMG10', 'EMG11', 'EMG12', 'EMG13', 'EMG14', 'EMG15', 'EMG16', 'unused7', 'unused8', 'blank2']
# df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused4', 'unused5', 'unused6', 'unused7', 'unused8', 'blank2']
# df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused7', 'unused8', 'blank2']
df2 = df.drop(['frames', 'subframes', 'blank', 'unused7', 'unused8', 'blank2'], axis = 1)
#print(df2)
df2 = df2.astype(np.float)
print(len(df))
hor = np.arange(0, (len(df)-0.5)/samplerate, 1/samplerate)  #getting the time domain in seconds
emg1 = df2.EMG8
emg2 = df2.EMG9
emg3 = df2.EMG11

# print(len(hor))
# plt.figure(1)
# plt.plot(hor, emg1)
# plt.title('GastMed')
# plt.figure(2)
# plt.plot(hor, emg2)
# plt.title('GastLat')
# plt.figure(3)
# plt.plot(hor, emg3)
# plt.title('VastLat')
# plt.show()
# exit()

cutoff_freq = 20  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='highpass', analog=False)
emg1high = signal.filtfilt(b, a, emg1)
emg2high = signal.filtfilt(b, a, emg2)
emg3high = signal.filtfilt(b, a, emg3)

cutoff_freq = 400  # ~500 Hz according to the emg book
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='lowpass', analog=False)
emg1filt = signal.filtfilt(b, a, emg1high)
emg2filt = signal.filtfilt(b, a, emg2high)
emg3filt = signal.filtfilt(b, a, emg3high)
print('yes')

freq12, t12, power12 = stft(emg1filt, samplerate, nperseg=512, noverlap=511)
freq13, t13, power13 = stft(emg2filt, samplerate, nperseg=512, noverlap=511)
freq14, t14, power14 = stft(emg3filt, samplerate, nperseg=512, noverlap=511)
plt.figure(1)
plt.pcolormesh(t12, freq12, np.abs(power12))
plt.figure(2)
plt.pcolormesh(t13, freq13, np.abs(power13))
plt.figure(3)
plt.pcolormesh(t14, freq14, np.abs(power14))
plt.show()