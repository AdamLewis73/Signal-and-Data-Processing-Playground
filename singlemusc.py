### CSV Extraction Code with help from: Jim Todd of Stack Overflow
from scipy import fftpack, signal
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
import scipy.signal as sps
temp = []
samplerate = 1500
nyq = samplerate*0.5


# with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Base_Noise.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Isometric.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Kinetic01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Kinetic02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Kinetic03.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Kinetic04.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR02.csv', 'r') as csvfile:       #Nathan Arm Iso
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR03.csv', 'r') as csvfile:     #Emily Arm Iso
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR04.csv', 'r') as csvfile:     #Adam Arm Iso
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR05.csv', 'r') as csvfile:     #Nathan Calf Raise Iso
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR06.csv', 'r') as csvfile:     #Emily Calf Raise Iso
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR07.csv', 'r') as csvfile:     #Nathan Arm Dyn Set 1
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR08.csv', 'r') as csvfile:     #Emily Arm Dyn Set 1
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR09.csv', 'r') as csvfile:     #Adam Arm Dyn Set 1
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR10.csv', 'r') as csvfile:     #Nathan Arm Dyn Set 2
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR11.csv', 'r') as csvfile:     #Emily Arm Dyn Set 2
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR12.csv', 'r') as csvfile:     #Adam Arm Dyn Set 2
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR13.csv', 'r') as csvfile:     #Nathan Arm Dyn Set 3
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR14.csv', 'r') as csvfile:     #Emily Arm Dyn Set 3
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR15.csv', 'r') as csvfile:     #Adam Arm Dyn Set 3
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR16.csv', 'r') as csvfile:     #Nathan Calf Raise Dyn Set 1
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR17.csv', 'r') as csvfile:     #Emily Calf Raise Dyn Set 1
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR18.csv', 'r') as csvfile:     #Nathan Calf Raise Dyn Set 2
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR19.csv', 'r') as csvfile:     #Emily Calf Raise Dyn Set 2


with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg_Fourth/Adam_Biodex_Leg4_TR02.csv', 'r') as csvfile:

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
df.columns = ['frames', 'subframes', 'blank', 'NU1', 'NU2', 'NatTA', 'NU3', 'NatBic', 'NatTri', 'NatGastLat', 'NatGastMed', 'EmilyBic', 'EmilyTri', 'EmilyGastMed', 'EmilyGastLat', 'EmilyTA', 'AdamBic', 'AdamTri', 'unused6', 'unused7', 'unused8', 'blank2']
# df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused4', 'unused5', 'unused6', 'unused7', 'unused8', 'blank2']
# df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused7', 'unused8', 'blank2']
df2 = df.drop(['frames', 'subframes', 'blank', 'unused7', 'unused8', 'blank2'], axis = 1)
#print(df2)
df2 = df2.astype(np.float)
print(len(df))
hor = np.arange(0, (len(df)-0.5)/samplerate, 1/samplerate)  #getting the time domain in seconds
emg1 = df2.NU1
print(len(hor))
plt.figure(1)
plt.plot(hor, emg1)
plt.title('Biceps')

cutoff_freq = 20  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='highpass', analog=False)
emg1high = signal.filtfilt(b, a, emg1)

cutoff_freq = 400  # ~500 Hz according to the emg book
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='lowpass', analog=False)
emg1filt = signal.filtfilt(b, a, emg1high)
####  INSERT ARRAY SPLICING HERE  ####
# #
# emg1temp = emg1filt[:33495]
# emg2temp = emg1filt[34305:38250]
# emg3temp = emg1filt[39495:86700]
# emg4temp = emg1filt[88200:307245]
# emg5temp = emg1filt[308970:]
# # emg6temp = emg1filt[362400:]
# # # emg7temp = emg1filt[:]
# # # emg8temp = emg1filt[:]
# # #
# emg1filt = np.concatenate([emg1temp, emg2temp, emg3temp, emg4temp, emg5temp])
# hor = np.arange(0, (len(emg1filt)-0.5)/samplerate, 1/samplerate)
# emg1filt = emg1filt[440000:460000]
# hor = hor[440000:460000]
####    END SPLICING CODE   ####
# emg1filt = emg1filt[:12500]
# hor = hor[:12500]
# emg1filt = emg1filt[12501:25000]
# hor = hor[12501:25000]
# emg1filt = emg1filt[25001:37500]
# hor = hor[25001:37500]
# emg1filt = emg1filt[37501:50000]
# hor = hor[37501:50000]
# emg1filt = emg1filt[50001:62500]
# hor = hor[50001:62500]
# emg1filt = emg1filt[62501:75000]
# hor = hor[62501:75000]
# emg1filt = emg1filt[75001:87500]
# hor = hor[75001:87500]
# emg1filt = emg1filt[87501:100000]
# hor = hor[87501:100000]
# emg1filt = emg1filt[100001:112500]
# hor = hor[100001:112500]
# emg1filt = emg1filt[112501:]
# hor = hor[112501:]
# #
# print(len(emg1filt))
# n=7
# m=n+1
#
# emg1filt = emg1filt[30001*n:m*30000]
# hor = hor[30001*n:m*30000]
#
# print(len(emg1filt))
# n=0
# m=n+1
#
# emg1filt = emg1filt[15001*n:m*15000]
# hor = hor[15001*n:m*15000]


plt.figure(2)
plt.plot(hor, emg1filt)
plt.title('Biceps Filtered')

freq1, power_spec1 = signal.periodogram(emg1filt, samplerate)

meantest1 = sum(freq1*power_spec1)/sum(power_spec1)
print('Biceps mean frequency is: ' + str(round(meantest1, 2))+' Hz')

plt.figure(3)
plt.semilogy(freq1, power_spec1)
plt.ylim([1e-17, 1e-7])
plt.title('Biceps: ' + str(round(meantest1, 2))+' Hz')

plt.figure(4)
f1,t1, Sxx1 = sps.spectrogram(emg1filt,samplerate,('hamming'),128,0, scaling= 'density')
plt.pcolormesh(t1,f1,Sxx1)
plt.ylim((0,300))
plt.title('Biceps')

###ISO POWER###
# print(Sxx1.shape)
# print(len(Sxx1))
# isopowavg = np.average(Sxx1[:20,:], axis = 1)
# # print(isopowavg.shape)
# for i in range(0,20):
#     print(isopowavg[i])

# exit()
###################

plt.figure(5)
plt.specgram(emg1filt,Fs=samplerate,NFFT=128,noverlap=0)

maxes1 = np.amax(Sxx1, axis=0)
maxesind1 = np.argmax(Sxx1, axis=0)
fval1 = [f1[i] for i in maxesind1]

plt.show()

fig = plt.figure(6)
ax12 = fig.add_subplot(1,1,1, label = '1')
ax12.xaxis.tick_top()
ax12.yaxis.tick_right()
# print(min(maxes1))
plt.plot(t1,maxes1, color = 'red')
ax1 = fig.add_subplot(1,1,1, label = '2', frame_on = False)
plt.plot(t1,fval1)
plt.title('Biceps')

maxfiltind1 = [i for i,ele in enumerate(maxes1) if ele>(min(maxes1)*15)]
maxfilt1 = [ele for i,ele in enumerate(maxes1) if ele>(min(maxes1)*15)]  ##Test - change val as neede

freqchan = []
freqchan = [Sxx1[:,i] for i in maxfiltind1]
freqchan = np.reshape(freqchan, (len(f1), len(maxfiltind1)))
freqpow = freqchan[:20,:]
freqpow2 = freqpow.mean(axis=1)

# print(freqchan)
# print(freqpow)
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

print(freqpow2[0])
print(freqpow2[1])
print(freqpow2[2])
print(freqpow2[3])
print(freqpow2[4])
print(freqpow2[5])
print(freqpow2[6])
print(freqpow2[7])
print(freqpow2[8])
print(freqpow2[9])
print(freqpow2[10])
print(freqpow2[11])
print(freqpow2[12])
print(freqpow2[13])
print(freqpow2[14])
print(freqpow2[15])
print(freqpow2[16])
print(freqpow2[17])
print(freqpow2[18])
print(freqpow2[19])
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
# csvfile.close()
# plt.show()

#
# with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Base_Noise.csv', 'r') as csvfile2:
# # with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Isometric.csv', 'r') as csvfile:
# # with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Kinetic01.csv', 'r') as csvfile:
# # with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Kinetic02.csv', 'r') as csvfile:
# # with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Kinetic03.csv', 'r') as csvfile:
# # with open('C:/Users/sword/Anaconda3/envs/exceltest/SingMusc/RBic01_Kinetic04.csv', 'r') as csvfile:
#
#     csvreader = csv.reader(csvfile2, delimiter=',')
#     for row in csvreader:
#         if csvreader.line_num == 3:
#             temp.append(row)
#         if csvreader.line_num >= 6:
#             if row:
#                 temp.append(row)
#             else:
#                 break
#
# df3 = pd.DataFrame(temp)  # turns the array into a dataframe
# df3.columns = df3.iloc[0]  # sets the column names as the first row
# df3 = df3.drop(0)  # drops the first row since it is now a duplicate of the column names
# df3.reindex(df3.index.drop(1))
# df3.reset_index(drop=True, inplace=True)
# #print(df)
# df3.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused4', 'unused5', 'unused6', 'unused7', 'unused8', 'blank2']
# # df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused7', 'unused8', 'blank2']
# df4 = df3.drop(['frames', 'subframes', 'blank', 'unused7', 'unused8', 'blank2'], axis = 1)
# # print(df4.RGastLat)
# # df4 = df4.astype(np.float,copy=True)
# # df4 = df4.convert_objects(convert_numeric=True)
# # print(len(df))
# hor2 = np.arange(0, (len(df3)-0.5)/samplerate, 1/samplerate)  #getting the time domain in seconds
# emg2 = df4.RRecFem
# # emg2 = pd.to_numeric(emg2, downcast = 'float')
# # print(emg2[0])
# # emg3 = emg2.astype(np.float)
#
# cutoff_freq = 20  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
# cut = cutoff_freq/nyq
# b, a = signal.butter(5, cut, btype='highpass', analog=False)
# emg2high = signal.filtfilt(b, a, emg2)
#
# cutoff_freq = 400  # ~500 Hz according to the emg book
# cut = cutoff_freq/nyq
# b, a = signal.butter(5, cut, btype='lowpass', analog=False)
# emg2filt = signal.filtfilt(b, a, emg2high)
#
# freq2, power_spec2 = signal.periodogram(emg2filt, samplerate)
#
#
# fig = plt.figure(8)
# ax12 = fig.add_subplot(1,1,1, label = '1')
# # print(min(maxes1))
# plt.semilogy(freq1, power_spec1)
# plt.ylim([1e-17, 1e-7])
# ax1 = fig.add_subplot(1,1,1, label = '2', frame_on = False)
# plt.semilogy(freq2, power_spec2, color = 'red')
# plt.ylim([1e-17, 1e-7])
# plt.title('Biceps: ' + str(round(meantest1, 2))+' Hz')
#
# plt.show()





##  Nathan Bicep Iso SPLICE ##
# emg1temp = emg1filt[:185400]
# hortemp  = hor[:185400]
# emg2temp = emg1filt[186390:319725]
# hor2temp = hor[186390:319725]
# # emg3temp = emg1filt[319725:320580]
# # hor3temp = hor[319725:320580]
# emg4temp = emg1filt[320580:336375]
# hor4temp = hor[320580:336375]
# # emg5temp = emg1filt[336375:337725]
# # hor5temp = hor[336375:337725]
# emg6temp = emg1filt[337725:]
# hor6temp = hor[337725:]
#
# emg1filt = np.concatenate([emg1temp, emg2temp, emg4temp, emg6temp])
# hor = np.concatenate([hortemp, hor2temp, hor4temp, hor6temp])

##  Nathan Tricep Iso SPLICE ##
# emg1temp = emg1filt[:210000]
# hor1temp  = hor[:210000]
# emg2temp = emg1filt[210900:243750]
# hor2temp = hor[210900:243750]
# emg3temp = emg1filt[244740:]
# hor3temp = hor[244740:]
# emg1filt = np.concatenate([emg1temp, emg2temp, emg3temp])
# hor = np.concatenate([hor1temp, hor2temp, hor3temp])

##  Emily Bicep Iso SPLICE ##
# emg1temp = emg1filt[:68760]
# hor1temp  = hor[:68760]
# emg2temp = emg1filt[69000:]
# hor2temp = hor[69000:]
# emg1filt = np.concatenate([emg1temp, emg2temp])
# hor = np.concatenate([hor1temp, hor2temp])

##  Adam Tricep Iso SPLICE ##
# emg1temp = emg1filt[:234780]
# emg2temp = emg1filt[235815:640500]
# emg3temp = emg1filt[650415:]
# #
# emg1filt = np.concatenate([emg1temp, emg2temp, emg3temp]).


##  Nathan GastMed Iso SPLICE ##
# emg1temp = emg1filt[:253275]
# emg2temp = emg1filt[254175:]
#
# emg1filt = np.concatenate([emg1temp, emg2temp])
# hor = np.arange(0, (len(emg1filt)-0.5)/samplerate, 1/samplerate)


##  Emily TA Iso SPLICE ##
# emg1temp = emg1filt[:260100]
# emg2temp = emg1filt[278400:331050]
# emg3temp = emg1filt[332400:1006125]
# emg4temp = emg1filt[1006425:]
#
# #
# emg1filt = np.concatenate([emg1temp, emg2temp, emg3temp, emg4temp])
# hor = np.arange(0, (len(emg1filt)-0.5)/samplerate, 1/samplerate)