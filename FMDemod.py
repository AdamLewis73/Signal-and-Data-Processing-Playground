### CSV Extraction Code with help from: Jim Todd of Stack Overflow
from scipy import fftpack, signal
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
import pylab as pl
import scipy.signal.signaltools as sigtool
import scipy.signal as signal
from numpy.random import sample
import scipy.signal as sps
temp = []
samplerate = 1000
nyq = samplerate*0.5


##################################
###   \\\ CSV EXTRACTION ///   ###
##################################

# with open('C:/Users/sword/Anaconda3/envs/exceltest/ZackTest/RF_SubjP01_Run_7MPH_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/ZackTest/RF_SubjP01_Run_7MPH_TR02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/ZackTest/RF_SubjP01_Run_7MPH_TR03.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/ZackTest/RF_SubjP01_Run_7MPH_TR04.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/ZackTest/RF_SubjP01_Run_7MPH_TR05.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/ZackTest/RF_SubjP01_Run_7MPH_TR06.csv', 'r') as csvfile:
with open('C:/Users/sword/Anaconda3/envs/exceltest/ZackTest/RF_SubjP01_Run_7MPH_TR07.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/ZackTest/RF_SubjP01_Run_7MPH_TR08.csv', 'r') as csvfile:


# with open('C:/Users/sword/Anaconda3/envs/exceltest/LukeTest/RF_Subj01_Jog_6MPH_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/LukeTest/RF_Subj01_Jog_6MPH_TR02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/LukeTest/RF_Subj01_Jog_6MPH_TR03.csv', 'r') as csvfile:

#with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_MVIC_PreGAS_TR01.csv', 'r') as csvfile:
#with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_MVIC_PreBiF_TR01.csv', 'r') as csvfile:
#with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_MVIC_PreRF_TR01.csv', 'r') as csvfile:
#with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_MVIC_PreTA_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR03.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR04.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR05.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR06.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR07.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR08.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR09.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR10.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR11.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR12.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue5_TR13.csv', 'r') as csvfile:
#with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Post5_TR02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue10_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue10_TR02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue10_TR03.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue10_TR04.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue10_TR05.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_Fatigue10_TR06.csv', 'r') as csvfile:

    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        if csvreader.line_num == 3:
            temp.append(row)
        if csvreader.line_num >= 6:
            if row:
                temp.append(row)
            else:
                break


######################################
###   \\\ DATAFRAME CREATION ///   ###
######################################


df = pd.DataFrame(temp)  # turns the array into a dataframe
df.columns = df.iloc[0]  # sets the column names as the first row
df = df.drop(0)  # drops the first row since it is now a duplicate of the column names
df.reindex(df.index.drop(1))
df.reset_index(drop=True, inplace=True)
#print(df)
# df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused4', 'unused5', 'unused6', 'unused7', 'unused8', 'blank2']
df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused7', 'unused8', 'blank2']
df2 = df.drop(['frames', 'subframes', 'blank', 'unused7', 'unused8', 'blank2'], axis = 1)
print(df2)
df2 = df2.astype(np.float)
#print(len(df))
hor = np.arange(0, len(df)/samplerate, 1/samplerate)  #getting the time domain in seconds
emg1 = df2.RGastLat
emg2 = df2.LGastLat
emg3 = df2.RTibAnt
emg4 = df2.LTibAnt
# emg5 = df2.RBicFem
# emg6 = df2.LBicFem
# emg7 = df2.RRecFem
# emg8 = df2.LRecFem
# emg9 = df2.RGastMed
# emg10 = df2.LGastMed
# emg1 = emg1.astype(np.float)
# print(emg1)


################################
###   \\\ PLOT RAW EMG ///   ###
################################


plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(hor, emg1)
plt.title('RGastMed')
plt.subplot(2, 2, 2)
plt.plot(hor, emg2)
plt.title('RVastLat')
plt.subplot(2, 2, 3)
plt.plot(hor, emg3)
plt.title('RBicFem')
plt.subplot(2, 2, 4)
plt.plot(hor, emg4)
plt.title('RRecFem')
# plt.subplot(3, 4, 5)
# plt.plot(hor, emg5)
# plt.title('RBicFem')
# plt.subplot(3, 4, 6)
# plt.plot(hor, emg6)
# plt.title('LBicFem')
# plt.subplot(3, 4, 7)
# plt.plot(hor, emg7)
# plt.title('RRecFem')
# plt.subplot(3, 4, 8)
# plt.plot(hor, emg8)
# plt.title('LRecFem')
# plt.subplot(3, 4, 9)
# plt.plot(hor, emg9)
# plt.title('RGastMed')
# plt.subplot(3, 4, 10)
# plt.plot(hor, emg10)
# plt.title('LGastMed')


##################################
###   \\\ FILTER SECTION ///   ###
##################################


cutoff_freq = 20  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='highpass', analog=False)
emg1high = signal.filtfilt(b, a, emg1)
emg2high = signal.filtfilt(b, a, emg2)
emg3high = signal.filtfilt(b, a, emg3)
emg4high = signal.filtfilt(b, a, emg4)
# emg5high = signal.filtfilt(b, a, emg5)
# emg6high = signal.filtfilt(b, a, emg6)
# emg7high = signal.filtfilt(b, a, emg7)
# emg8high = signal.filtfilt(b, a, emg8)
# emg9high = signal.filtfilt(b, a, emg9)
# emg10high = signal.filtfilt(b, a, emg10)

cutoff_freq = 20  # ~500 Hz according to the emg book
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='lowpass', analog=False)
emg1filt = signal.filtfilt(b, a, emg1high)
emg2filt = signal.filtfilt(b, a, emg2high)
emg3filt = signal.filtfilt(b, a, emg3high)
emg4filt = signal.filtfilt(b, a, emg4high)
# emg5filt = signal.filtfilt(b, a, emg5high)
# emg6filt = signal.filtfilt(b, a, emg6high)
# emg7filt = signal.filtfilt(b, a, emg7high)
# emg8filt = signal.filtfilt(b, a, emg8high)
# emg9filt = signal.filtfilt(b, a, emg9high)
# emg10filt = signal.filtfilt(b, a, emg10high)

plt.figure(2)
plt.subplot(2, 2, 1)
plt.plot(hor, emg1filt)
plt.title('RGastMed Filtered')
plt.subplot(2, 2, 2)
plt.plot(hor, emg2filt)
plt.title('RVastLat Filtered')
plt.subplot(2, 2, 3)
plt.plot(hor, emg3filt)
plt.title('RBicFem Filtered')
plt.subplot(2, 2, 4)
plt.plot(hor, emg4filt)
plt.title('RRecFem Filtered')
# plt.subplot(3, 4, 5)
# plt.plot(hor, emg5filt)
# plt.title('RBicFem Filtered')
# plt.subplot(3, 4, 6)
# plt.plot(hor, emg6filt)
# plt.title('LBicFem Filtered')
# plt.subplot(3, 4, 7)
# plt.plot(hor, emg7filt)
# plt.title('RRecFem Filtered')
# plt.subplot(3, 4, 8)
# plt.plot(hor, emg8filt)
# plt.title('LRecFem Filtered')
# plt.subplot(3, 4, 9)
# plt.plot(hor, emg9filt)
# plt.title('RGastMed Filtered')
# plt.subplot(3, 4, 10)
# plt.plot(hor, emg10filt)
# plt.title('LGastMed Filtered')
# plt.show()

#####################################################
###   \\\ POWER SPECTRUM AND MEAN FREQUENCY ///   ###
#####################################################

freq1, power_spec1 = signal.periodogram(emg1filt, samplerate)
freq2, power_spec2 = signal.periodogram(emg2filt, samplerate)
freq3, power_spec3 = signal.periodogram(emg3filt, samplerate)
freq4, power_spec4 = signal.periodogram(emg4filt, samplerate)
# freq5, power_spec5 = signal.periodogram(emg5filt, samplerate)
# freq6, power_spec6 = signal.periodogram(emg6filt, samplerate)
# freq7, power_spec7 = signal.periodogram(emg7filt, samplerate)
# freq8, power_spec8 = signal.periodogram(emg8filt, samplerate)
# freq9, power_spec9 = signal.periodogram(emg9filt, samplerate)
# freq10, power_spec10 = signal.periodogram(emg10filt, samplerate)
meantest1 = sum(freq1*power_spec1)/sum(power_spec1)
print('RGastLat mean frequency is: ' + str(round(meantest1, 2))+' Hz')
meantest2 = sum(freq2*power_spec2)/sum(power_spec2)
print('LGastLat mean frequency is: ' + str(round(meantest2, 2))+' Hz')
meantest3 = sum(freq3*power_spec3)/sum(power_spec3)
print('RTibAnt mean frequency is: ' + str(round(meantest3, 2))+' Hz')
meantest4 = sum(freq4*power_spec4)/sum(power_spec4)
print('LTibAnt mean frequency is: ' + str(round(meantest4, 2))+' Hz')
# meantest5 = sum(freq5*power_spec5)/sum(power_spec5)
# print('RBicFem mean frequency is: ' + str(round(meantest5, 2))+' Hz')
# meantest6 = sum(freq6*power_spec6)/sum(power_spec6)
# print('LBicFem mean frequency is: ' + str(round(meantest6, 2))+' Hz')
# meantest7 = sum(freq7*power_spec7)/sum(power_spec7)
# print('RRecFem mean frequency is: ' + str(round(meantest7, 2))+' Hz')
# meantest8 = sum(freq8*power_spec8)/sum(power_spec8)
# print('LRecFem mean frequency is: ' + str(round(meantest8, 2))+' Hz')
# meantest9 = sum(freq9*power_spec9)/sum(power_spec9)
# print('RGastMed mean frequency is: ' + str(round(meantest9, 2))+' Hz')
# meantest10 = sum(freq10*power_spec10)/sum(power_spec10)
# print('LGastMed mean frequency is: ' + str(round(meantest10, 2))+' Hz')
plt.figure(3)
plt.semilogy(freq1, power_spec1)
plt.ylim([1e-17, 1e-8])
plt.semilogy(freq2, power_spec2)
plt.ylim([1e-17, 1e-8])
plt.semilogy(freq3, power_spec3)
plt.ylim([1e-17,1e-8])
plt.semilogy(freq4, power_spec4)
plt.ylim([1e-17,1e-8])
# plt.semilogy(freq5, power_spec5)
# plt.ylim([1e-17,1e-8])
# plt.semilogy(freq6, power_spec6)
# plt.ylim([1e-17,1e-8])
# plt.semilogy(freq7, power_spec7)
# plt.ylim([1e-17,1e-8])
# plt.semilogy(freq8, power_spec8)
# plt.ylim([1e-17,1e-8])
# plt.semilogy(freq9, power_spec9)
# plt.ylim([1e-17,1e-8])
# plt.semilogy(freq10, power_spec10)
# plt.ylim([1e-17,1e-8])
#plt.legend()

plt.figure(4)
plt.subplot(2, 2, 1)
plt.semilogy(freq1, power_spec1)
plt.ylim([1e-17, 1e-7])
plt.title('RGastMed: ' + str(round(meantest1, 2))+' Hz')
plt.subplot(2, 2, 2)
plt.semilogy(freq2, power_spec2)
plt.ylim([1e-17, 1e-7])
plt.title('RVastLat: ' + str(round(meantest2, 2))+' Hz')
plt.subplot(2, 2, 3)
plt.semilogy(freq3, power_spec3)
plt.ylim([1e-17,1e-7])
plt.title('RBicFem: ' + str(round(meantest3, 2))+' Hz')
plt.subplot(2, 2, 4)
plt.semilogy(freq4, power_spec4)
plt.ylim([1e-17,1e-7])
plt.title('RRecFem: ' + str(round(meantest4, 2))+' Hz')
# plt.subplot(3, 4, 5)
# plt.semilogy(freq5, power_spec5)
# plt.ylim([1e-17,1e-7])
# plt.title('RBicFem: ' + str(round(meantest5, 2))+' Hz')
# plt.subplot(3, 4, 6)
# plt.semilogy(freq6, power_spec6)
# plt.ylim([1e-17,1e-7])
# plt.title('LBicFem: ' + str(round(meantest6, 2))+' Hz')
# plt.subplot(3, 4, 7)
# plt.semilogy(freq7, power_spec7)
# plt.ylim([1e-17,1e-7])
# plt.title('RRecFem: ' + str(round(meantest7, 2))+' Hz')
# plt.subplot(3, 4, 8)
# plt.semilogy(freq8, power_spec8)
# plt.ylim([1e-17,1e-7])
# plt.title('LRecFem: ' + str(round(meantest8, 2))+' Hz')
# plt.subplot(3, 4, 9)
# plt.semilogy(freq9, power_spec9)
# plt.ylim([1e-17,1e-7])
# plt.title('RGastMed: ' + str(round(meantest9, 2))+' Hz')
# plt.subplot(3, 4, 10)
# plt.semilogy(freq10, power_spec10)
# plt.ylim([1e-17,1e-7])
# plt.title('LGastMed: ' + str(round(meantest10, 2))+' Hz')
#plt.show()

# emgfft = fftpack.fft(yfilt2, hor.size)
# emgfftabs = np.abs(emgfft)
# xf = fftpack.fftfreq(hor.size, (len(yfilt2) / samplerate) / samplerate) #finding the length in seconds divided by samples
#

################################
###   \\\ SPECTROGRAMS ///   ###
################################

plt.figure(5)
plt.subplot(2,2,1)
f1,t1, Sxx1 = sps.spectrogram(emg1filt,samplerate,('hamming'),128,0, scaling= 'density')
# Normalization technique? didn't seem to work
# max1 = np.amax(Sxx1)
# min1 = np.amin(Sxx1)
# print("Max: ", max1)
# print("Min: ", min1)
# print(np.shape(Sxx1))
# Sxx1norm = [((x-min1)/(max1-min1)) for x in Sxx1]
# Sxx1norm = np.reshape(Sxx1norm,(65,469))
# print(Sxx1[22,13])
# print(Sxx1norm[22,13])
plt.pcolormesh(t1,f1,Sxx1)
plt.ylim((0,300))
plt.title('RGastMed')

plt.subplot(2,2,2)
f2,t2, Sxx2 = sps.spectrogram(emg2filt,samplerate,('hamming'),128,0)
plt.pcolormesh(t2,f2,Sxx2)
plt.ylim((0,300))
plt.title('RVastLat')

plt.subplot(2,2,3)
f3,t3, Sxx3 = sps.spectrogram(emg3filt,samplerate,('hamming'),128,0)
plt.pcolormesh(t3,f3,Sxx3)
plt.ylim((0,300))
plt.title('RBicFem')

plt.subplot(2,2,4)
f4,t4, Sxx4 = sps.spectrogram(emg4filt,samplerate,('hamming'),128,0)
plt.pcolormesh(t4,f4,Sxx4)
plt.ylim((0,300))
plt.title('RRecFem')


# plt.subplot(3,4,5)
# f5,t5, Sxx5 = sps.spectrogram(emg5filt,samplerate,('hamming'),128,64)
# plt.pcolormesh(t5,f5,Sxx5)
# plt.ylim((0,300))
# plt.title('RBicFem')
#
# plt.subplot(3,4,6)
# f6,t6, Sxx6 = sps.spectrogram(emg6filt,samplerate,('hamming'),128,64)
# plt.pcolormesh(t6,f6,Sxx6)
# plt.ylim((0,300))
# plt.title('LBicFem')
#
#
# plt.subplot(3,4,7)
# f7,t7, Sxx7 = sps.spectrogram(emg7filt,samplerate,('hamming'),128,64)
# plt.pcolormesh(t7,f7,Sxx7)
# plt.ylim((0,300))
# plt.title('RRecFem')
#
#
# plt.subplot(3,4,8)
# f8,t8, Sxx8 = sps.spectrogram(emg8filt,samplerate,('hamming'),128,64)
# plt.pcolormesh(t8,f8,Sxx8)
# plt.ylim((0,300))
# plt.title('LRecFem')
#
#
# plt.subplot(3,4,9)
# f9,t9, Sxx9 = sps.spectrogram(emg9filt,samplerate,('hamming'),128,64)
# plt.pcolormesh(t9,f9,Sxx9)
# plt.ylim((0,300))
# plt.title('RGastMed')
#
#
# plt.subplot(3,4,10)
# f10,t10, Sxx10 = sps.spectrogram(emg10filt,samplerate,('hamming'),128,64)
# plt.pcolormesh(t10,f10,Sxx10)
# plt.ylim((0,300))
# plt.title('LGastMed')

# plt.figure(6)
# plt.specgram(emg1filt,Fs=samplerate,NFFT=128,noverlap=64)
# #plt.show()




maxes1 = np.amax(Sxx1, axis=0)
maxesind1 = np.argmax(Sxx1, axis=0)
fval1 = [f1[i] for i in maxesind1]
maxes2 = np.amax(Sxx2, axis=0)
maxesind2 = np.argmax(Sxx2, axis=0)
fval2 = [f2[i] for i in maxesind2]
maxes3 = np.amax(Sxx3, axis=0)
maxesind3 = np.argmax(Sxx3, axis=0)
fval3 = [f3[i] for i in maxesind3]
maxes4 = np.amax(Sxx4, axis=0)
maxesind4 = np.argmax(Sxx4, axis=0)
fval4 = [f4[i] for i in maxesind4]
# maxes5 = np.amax(Sxx5, axis=0)
# maxesind5 = np.argmax(Sxx5, axis=0)
# fval5 = [f5[i] for i in maxesind5]
# maxes6 = np.amax(Sxx6, axis=0)
# maxesind6 = np.argmax(Sxx6, axis=0)
# fval6 = [f6[i] for i in maxesind6]
# maxes7 = np.amax(Sxx7, axis=0)
# maxesind7 = np.argmax(Sxx7, axis=0)
# fval7 = [f7[i] for i in maxesind7]
# maxes8 = np.amax(Sxx8, axis=0)
# maxesind8 = np.argmax(Sxx8, axis=0)
# fval8 = [f8[i] for i in maxesind8]
# maxes9 = np.amax(Sxx9, axis=0)
# maxesind9 = np.argmax(Sxx9, axis=0)
# fval9 = [f9[i] for i in maxesind9]
# maxes10 = np.amax(Sxx10, axis=0)
# maxesind10 = np.argmax(Sxx10, axis=0)
# fval10 = [f10[i] for i in maxesind10]


# print(maxesind1)

fig = plt.figure(7)
# maxes1 = np.amax(Sxx1, axis=0)
# maxes = np.argmax(Sxx1, axis=0)
# fval = [f1[i] for i in maxes]
plt.subplot(2,2,1)
ax12 = fig.add_subplot(2,2,1, label = '1')
ax12.xaxis.tick_top()
ax12.yaxis.tick_right()
# print(min(maxes1))
plt.plot(t1,maxes1, color = 'red')

ax1 = fig.add_subplot(2,2,1, label = '2', frame_on = False)
plt.plot(t1,fval1)
plt.title('RGastMed')


plt.subplot(2,2,2)
ax2 = fig.add_subplot(2,2,2, label = '1')
plt.plot(t2,fval2)
ax22 = fig.add_subplot(2,2,2, label = '2', frame_on = False)
ax22.xaxis.tick_top()
ax22.yaxis.tick_right()
# print(min(maxes1))
plt.plot(t2,maxes2, color = 'red')
plt.title('RVastLat')

plt.subplot(2,2,3)
ax3 = fig.add_subplot(2,2,3, label = '1')
plt.plot(t3,fval3)
ax32 = fig.add_subplot(2,2,3, label = '2', frame_on = False)
ax32.xaxis.tick_top()
ax32.yaxis.tick_right()
# print(min(maxes1))
plt.plot(t3,maxes3, color = 'red')
plt.title('RBicFem')

plt.subplot(2,2,4)
ax4 = fig.add_subplot(2,2,4, label = '1')
plt.plot(t4,fval4)
ax42 = fig.add_subplot(2,2,4, label = '2', frame_on = False)
ax42.xaxis.tick_top()
ax42.yaxis.tick_right()
# print(min(maxes1))
plt.plot(t4,maxes4, color = 'red')
plt.title('RRecFem')

# plt.subplot(3,2,5)
# ax5 = fig.add_subplot(3,2,5, label = '1')
# plt.plot(t5,fval5)
# ax52 = fig.add_subplot(3,2,5, label = '2', frame_on = False)
# ax52.xaxis.tick_top()
# ax52.yaxis.tick_right()
# # print(min(maxes1))
# plt.plot(t5,maxes5, color = 'red')
# plt.title('RBicFem')
#
# fig = plt.figure(8)
# plt.subplot(3,2,1)
# ax6 = fig.add_subplot(3,2,1, label = '1')
# plt.plot(t6,fval6)
# ax62 = fig.add_subplot(3,2,1, label = '2', frame_on = False)
# ax62.xaxis.tick_top()
# ax62.yaxis.tick_right()
# # print(min(maxes1))
# plt.plot(t6,maxes6, color = 'red')
# plt.title('LBicFem')
#
# plt.subplot(3,2,2)
# ax7 = fig.add_subplot(3,2,2, label = '1')
# plt.plot(t7,fval7)
# ax72 = fig.add_subplot(3,2,2, label = '2', frame_on = False)
# ax72.xaxis.tick_top()
# ax72.yaxis.tick_right()
# # print(min(maxes1))
# plt.plot(t7,maxes7, color = 'red')
# plt.title('RRecFem')
#
# plt.subplot(3,2,3)
# ax8 = fig.add_subplot(3,2,3, label = '1')
# plt.plot(t8,fval8)
# ax82 = fig.add_subplot(3,2,3, label = '2', frame_on = False)
# ax82.xaxis.tick_top()
# ax82.yaxis.tick_right()
# # print(min(maxes1))
# plt.plot(t8,maxes8, color = 'red')
# plt.title('LRecFem')
#
# plt.subplot(3,2,4)
# ax9 = fig.add_subplot(3,2,4, label = '1')
# plt.plot(t9,fval9)
# ax92 = fig.add_subplot(3,2,4, label = '2', frame_on = False)
# ax92.xaxis.tick_top()
# ax92.yaxis.tick_right()
# # print(min(maxes1))
# plt.plot(t9,maxes9, color = 'red')
# plt.title('RGastMed')
#
# plt.subplot(3,2,5)
# ax10 = fig.add_subplot(3,2,5, label = '1')
# plt.plot(t10,fval10)
# ax102 = fig.add_subplot(3,2,5, label = '2', frame_on = False)
# ax102.xaxis.tick_top()
# ax102.yaxis.tick_right()
# # print(min(maxes1))
# plt.plot(t10,maxes10, color = 'red')
# plt.title('LGastMed')

# plt.show()

# maxfiltind = [i for i,ele in enumerate(maxes1) if ele>5e-11]
# maxfilt = [ele for i,ele in enumerate(maxes1) if ele>5e-11]  ##FF5T2
# maxfiltind = [i for i,ele in enumerate(maxes1) if ele>(min(maxes1)*15)]
# maxfilt = [ele for i,ele in enumerate(maxes1) if ele>(min(maxes1)*15)]  ##Test - change val as needed
# maxfiltind = [i for i,ele in enumerate(maxes1) if ele>5e-13]
# maxfilt = [ele for i,ele in enumerate(maxes1) if ele>5e-13]  ##FF5T13
# maxfiltind = [i for i,ele in enumerate(maxes1) if ele>3e-13]
# maxfilt = [ele for i,ele in enumerate(maxes1) if ele>3e-13]  ##FP5T2
# maxfiltind = [i for i,ele in enumerate(maxes1) if ele>1.1e-13]
# maxfilt = [ele for i,ele in enumerate(maxes1) if ele>1.1e-13]  ##FF10T1
# maxfiltind = [i for i,ele in enumerate(maxes1) if ele>1.1e-13]
# maxfilt = [ele for i,ele in enumerate(maxes1) if ele>1.1e-13]  ##FF10T2
# maxfiltind = [i for i,ele in enumerate(maxes1) if ele>1.1e-13]
# maxfilt = [ele for i,ele in enumerate(maxes1) if ele>1.1e-13]  ##FF10T3
# maxfiltind = [i for i,ele in enumerate(maxes1) if ele>1.5e-13]
# maxfilt = [ele for i,ele in enumerate(maxes1) if ele>1.5e-13] ##FF10T4
# maxfiltind = [i for i,ele in enumerate(maxes1) if ele>7.e-14]
# maxfilt = [ele for i,ele in enumerate(maxes1) if ele>7.e-14]  ##FF10T5
# maxfiltind = [i for i,ele in enumerate(maxes1) if ele>1.1e-13]
# maxfilt = [ele for i,ele in enumerate(maxes1) if ele>1.1e-13]  ##FF10T6

maxfiltind1 = [i for i,ele in enumerate(maxes1) if ele>(min(maxes1)*15)]
maxfilt1 = [ele for i,ele in enumerate(maxes1) if ele>(min(maxes1)*15)]  ##Test - change val as needed
maxfiltind2 = [i for i,ele in enumerate(maxes2) if ele>(min(maxes2)*15)]
maxfilt2 = [ele for i,ele in enumerate(maxes2) if ele>(min(maxes2)*15)]  ##Test - change val as needed
maxfiltind3 = [i for i,ele in enumerate(maxes3) if ele>(min(maxes3)*15)]
maxfilt3 = [ele for i,ele in enumerate(maxes3) if ele>(min(maxes3)*15)]  ##Test - change val as needed
maxfiltind4 = [i for i,ele in enumerate(maxes4) if ele>(min(maxes4)*15)]
maxfilt4 = [ele for i,ele in enumerate(maxes4) if ele>(min(maxes4)*15)]  ##Test - change val as needed
# maxfiltind5 = [i for i,ele in enumerate(maxes5) if ele>(min(maxes5)*15)]
# maxfilt5 = [ele for i,ele in enumerate(maxes5) if ele>(min(maxes5)*15)]  ##Test - change val as needed
# maxfiltind6 = [i for i,ele in enumerate(maxes6) if ele>(min(maxes6)*15)]
# maxfilt6 = [ele for i,ele in enumerate(maxes6) if ele>(min(maxes6)*15)]  ##Test - change val as needed
# maxfiltind7 = [i for i,ele in enumerate(maxes7) if ele>(min(maxes7)*15)]
# maxfilt7 = [ele for i,ele in enumerate(maxes7) if ele>(min(maxes7)*15)]  ##Test - change val as needed
# maxfiltind8 = [i for i,ele in enumerate(maxes8) if ele>(min(maxes8)*15)]
# maxfilt8 = [ele for i,ele in enumerate(maxes8) if ele>(min(maxes8)*15)]  ##Test - change val as needed
# maxfiltind9 = [i for i,ele in enumerate(maxes9) if ele>(min(maxes9)*15)]
# maxfilt9 = [ele for i,ele in enumerate(maxes9) if ele>(min(maxes9)*15)]  ##Test - change val as needed
# maxfiltind10 = [i for i,ele in enumerate(maxes10) if ele>(min(maxes10)*15)]
# maxfilt10 = [ele for i,ele in enumerate(maxes10) if ele>(min(maxes10)*15)]  ##Test - change val as needed


freqchan = []
freqchan = [Sxx1[:,i] for i in maxfiltind1]
freqchan = np.reshape(freqchan, (len(f1), len(maxfiltind1)))
freqpow = freqchan[:20,:]
freqpow2 = freqpow.mean(axis=1)

# print(freqpow2)
# print(freqpow2.shape)
print("         <  7.8125 Hz: ", freqpow2[0])
print("7.8125  -   15.625 Hz: ", freqpow2[1])
print("15.625 -   23.4375 Hz: ", freqpow2[2])
print("23.4375  -   31.25 Hz: ", freqpow2[3])
print("31.25  -   39.0625 Hz: ", freqpow2[4])
print("39.0625  -  46.875 Hz: ", freqpow2[5])
print("46.875  -  54.6875 Hz: ", freqpow2[6])
print("54.6875    -  62.5 Hz: ", freqpow2[7])
print("62.5    -  70.3125 Hz: ", freqpow2[8])
print("70.3125  -  78.125 Hz: ", freqpow2[9])
print("78.125  -  85.9375 Hz: ", freqpow2[10])
print("85.9375   -  93.75 Hz: ", freqpow2[11])
print("93.75  -  101.5625 Hz: ", freqpow2[12])
print("101.5625 - 109.375 Hz: ", freqpow2[13])
print("109.375 - 117.1875 Hz: ", freqpow2[14])
print("117.1875   -   125 Hz: ", freqpow2[15])
print("125   -   132.8125 Hz: ", freqpow2[16])
print("132.8125 - 140.625 Hz: ", freqpow2[17])
print("140.625 - 148.4375 Hz: ", freqpow2[18])
print("148.4375 -  156.25 Hz: ", freqpow2[19])
# plt.show()



saveind = []
saveind = np.empty(shape=(0,2))
for index,k in np.ndenumerate(Sxx1):
    for val in maxfilt1:
        if k == val:
            saveind = np.append(saveind, index)
saveind = np.reshape(saveind,(len(saveind)//2,2))
saveind = saveind[saveind[:,1].argsort()]
saveind = np.delete(saveind,1,1)
saveind = saveind.astype(int)

fval1 = [f1[i] for i in saveind]
fval1 = np.concatenate(fval1,axis = 0)
avg1 = np.average(fval1)
print('RGastLat Avg: ', avg1)


saveind2 = []
saveind2 = np.empty(shape=(0,2))
for index,k in np.ndenumerate(Sxx2):
    for val in maxfilt2:
        if k == val:
            saveind2 = np.append(saveind2, index)

saveind2 = np.reshape(saveind2,(len(saveind2)//2,2))
saveind2 = saveind2[saveind2[:,1].argsort()]
saveind2 = np.delete(saveind2,1,1)
saveind2 = saveind2.astype(int)

fval2 = [f2[i] for i in saveind2]
fval2 = np.concatenate(fval2,axis = 0)

avg2 = np.average(fval2)
print('RVastLat Avg: ', avg2)


saveind3 = []
saveind3 = np.empty(shape=(0, 2))
for index, k in np.ndenumerate(Sxx3):
    for val in maxfilt3:
        if k == val:
            saveind3 = np.append(saveind3, index)

saveind3 = np.reshape(saveind3, (len(saveind3) // 2, 2))
saveind3 = saveind3[saveind3[:, 1].argsort()]
saveind3 = np.delete(saveind3, 1, 1)
saveind3 = saveind3.astype(int)

fval3 = [f3[i] for i in saveind3]
fval3 = np.concatenate(fval3, axis=0)

avg3 = np.average(fval3)
print('RBicFem Avg: ', avg3)

saveind4 = []
saveind4 = np.empty(shape=(0, 2))
for index, k in np.ndenumerate(Sxx4):
    for val in maxfilt4:
        if k == val:
            saveind4 = np.append(saveind4, index)

saveind4 = np.reshape(saveind4, (len(saveind4) // 2, 2))
saveind4 = saveind4[saveind4[:, 1].argsort()]
saveind4 = np.delete(saveind4, 1, 1)
saveind4 = saveind4.astype(int)

fval4 = [f4[i] for i in saveind4]
fval4 = np.concatenate(fval4, axis=0)

avg4 = np.average(fval4)
print('RRecFem Avg: ', avg4)

plt.show()








# Fc = 100       #simulate a carrier frequency of 1kHz
# Fbit = 50       #simulated bitrate of data
# Fdev = 500      #frequency deviation, make higher than bitrate
# N = 64          #how many bits to send
# A = 1           #transmitted signal amplitude
# Fs = 1000      #sampling frequency for the simulator, must be higher than twice the carrier frequency
# A_n = 0.10      #noise peak amplitude
# N_prntbits = 10 #number of bits to print in plots
#
# y_diff = np.diff(emg1filt,1)

# """
# Envelope detector + low-pass filter
# """
#create an envelope detector and then low-pass filter
# y_env = np.abs(sigtool.hilbert(y_diff))
# h=signal.firwin( numtaps=100, cutoff=Fbit*2, nyq=Fs/2)
# y_filtered=signal.lfilter( h, 1.0, y_env)
# #view the data after adding noise
# N_FFT = float(len(y_filtered))
# f = np.arange(0,Fs/2,Fs/N_FFT)
# w = np.hanning(len(y_filtered))
# y_f = np.fft.fft(np.multiply(y_filtered,w))
# y_f = 10*np.log10(np.abs(y_f[0:N_FFT//2]//N_FFT))
# plt.figure(3)
# pl.subplot(3,1,1)
# pl.plot(t[0:Fs*N_prntbits/Fbit],m[0:Fs*N_prntbits/Fbit])
# pl.xlabel('Time (s)')
# pl.ylabel('Frequency (Hz)')
# pl.title('Original VCO output vs. time')
# pl.grid(True)
# pl.subplot(3,1,2)
# pl.plot(t[0:Fs*N_prntbits/Fbit],np.abs(y[0:Fs*N_prntbits/Fbit]),'b')
# pl.plot(t[0:Fs*N_prntbits/Fbit],y_filtered[0:Fs*N_prntbits/Fbit],'g',linewidth=3.0)
# pl.xlabel('Time (s)')
# pl.ylabel('Amplitude (V)')
# pl.title('Filtered signal and unfiltered signal vs. time')
# pl.grid(True)
# pl.subplot(3,1,3)
# pl.plot(f[0:(Fc+Fdev*2)*N_FFT/Fs],y_f[0:(Fc+Fdev*2)*N_FFT/Fs])
# pl.xlabel('Frequency (Hz)')
# pl.ylabel('Amplitude (dB)')
# pl.title('Spectrum')
# pl.grid(True)
# pl.tight_layout()
# pl.show()
