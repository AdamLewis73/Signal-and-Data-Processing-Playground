### CSV Extraction Code with help from: Jim Todd of Stack Overflow
from scipy import fftpack, signal, integrate
from matplotlib import pyplot as plt
from scipy.signal import hilbert
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
with open('C:/Users/sword/PycharmProjects/excelimport/PT10/Free/RF2_Subj10_Free_Fatigue_TR10.csv','r') as csvfile:


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

print(len(hor))
plt.figure(1)
plt.plot(hor, emg1)
plt.title('GastMed')
plt.figure(2)
plt.plot(hor, emg2)
plt.title('GastLat')
plt.figure(3)
plt.plot(hor, emg3)
plt.title('VastLat')
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
####  INSERT ARRAY SPLICING HERE  ####

#Subj1 Prebiodex
#Pulse 1 = [8259:12715]
#Pulse 2 = [17026:22403]
#Pulse 3 = [26664:31266]
#Pulse 4 = [36303:40904]
#Pulse 5 = [45120:51120]
#Pulse 6 = [56730:61290]

#Subj1 Postbiodex
#Pulse 1 = [4209:10995]
#Pulse 2 = [15810:21000]
#Pulse 3 = [25605:30540]
#Pulse 4 = [35805:40860]
#Pulse 5 = [45735:50475]
#Pulse 6 = [55530:60525]

#Subj2 Prebiodex
#Pulse 1 = [15900:19950]
#Pulse 2 = [24375:28860]
#Pulse 3 = [33720:38970]
#Pulse 4 = [43950:49200]
#Pulse 5 = [54000:58800]
#Pulse 6 = [63300:68775]

#Subj2 Postbiodex
#Pulse 1 = [16500:20895]
#Pulse 2 = [25800:30870]
#Pulse 3 = [35610:40020]
#Pulse 4 = [45765:50505]
#Pulse 5 = [56100:60840]
#Pulse 6 = [66420:71325]

#Subj3 Prebiodex
#Pulse 1 = [4320:11475]
#Pulse 2 = [15315:20625]
#Pulse 3 = [24255:29340]
#Pulse 4 = [32700:37170]
#Pulse 5 = [41250:46800]
#Pulse 6 = [50925:56415]

#Subj3 Postbiodex
#Pulse 1 = [1725:8355]
#Pulse 2 = [13200:18450]
#Pulse 3 = [22920:27810]
#Pulse 4 = [32415:37050]
#Pulse 5 = [41910:46500]
#Pulse 6 = [52125:58125]

#Subj4 Prebiodex
#Pulse 1 = []
#Pulse 2 = []
#Pulse 3 = []
#Pulse 4 = []
#Pulse 5 = []
#Pulse 6 = []

#Subj2 Postbiodex
#Pulse 1 = []
#Pulse 2 = []
#Pulse 3 = []
#Pulse 4 = []
#Pulse 5 = []
#Pulse 6 = []

#Subj2 Postbiodex
#Pulse 1 = []
#Pulse 2 = []
#Pulse 3 = []
#Pulse 4 = []
#Pulse 5 = []
#Pulse 6 = []

#Subj2 Postbiodex
#Pulse 1 = []
#Pulse 2 = []
#Pulse 3 = []
#Pulse 4 = []
#Pulse 5 = []
#Pulse 6 = []

#Subj2 Postbiodex
#Pulse 1 = []
#Pulse 2 = []
#Pulse 3 = []
#Pulse 4 = []
#Pulse 5 = []
#Pulse 6 = []

#Subj2 Postbiodex
#Pulse 1 = []
#Pulse 2 = []
#Pulse 3 = []
#Pulse 4 = []
#Pulse 5 = []
#Pulse 6 = []

# emg1filt = emg1filt[55530:60525]
# emg2filt = emg2filt[55530:60525]
# emg3filt = emg3filt[55530:60525]
# hor = hor[55530:60525]

plt.figure(5)
# freq12, power12 = signal.welch(emg1filt,samplerate,nperseg=512)

freq12, t12, power12 = signal.stft(emg1filt, samplerate, nperseg=512)
power12 = np.real(power12)
print(len(power12[1,]))
# exit()

for i in range(len(power12[1,])):
    # x12 = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(power12[i,])
    # df = pd.DataFrame(x_scaled)
plt.semilogy(freq12,x_scaled)
plt.show()
print(freq12)
print(t12)
print(power12)
print(len(freq12))
print(power12.shape)
print(len(t12))
# print(power12[256,67])
exit()
trash = 0
trash2 = 0
freqtrash = []
for i in range(len(t12)):
    trash = 0
    trash2 = 0
    for j in range(len(freq12)):
        # print(power12[j,i])
        # print(freq12[j])
        # print(power12[j, i]*freq12[j])
        trash = trash + (power12[j,i]*freq12[j])
        trash2 = trash2 + power12[j,i]
        # print(trash)
        # print(trash2)
    # print(trash)
    # print(trash2)
    trashmean = trash/trash2
    freqtrash.append(trashmean)
# print(freqtrash)
# exit()
# exit()
plt.figure(8)
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.pcolormesh(t12, freq12, np.abs(power12))
ax2.plot(t12, freqtrash)

# plt.show()

for i in range(0,3):
    # plt.figure(i+1)
    # plt.plot(hor, emg1filt)
    print('EMG ' + str(i))
    if i == 0:
        freq, power_spec = signal.periodogram(emg1filt, samplerate)
    elif i == 1:
        freq, power_spec = signal.periodogram(emg2filt, samplerate)
    elif i == 2:
        freq, power_spec = signal.periodogram(emg3filt, samplerate)

    # print('Biceps mean frequency is: ' + str(round(meantest1, 2))+' Hz')
    # lowflag = 0
    # breakflag = 0
    # for c in range(len(freq)):
    #     if abs(freq[c]) > 10 and lowflag == 0:
    #         fullpulse_lowfreqbound = c
    #         lowflag = 1
    #     if abs(freq[c]) > 400:
    #         fullpulse_uppfreqbound = c
    #         break
    #
    # spec_mom0 = 0
    # spec_mom1 = 0
    # spec_mom2 = 0
    # spec_mom5 = 0
    #
    # for k in range(fullpulse_lowfreqbound, fullpulse_uppfreqbound):
    #     spec_mom0 = spec_mom0 + (math.pow(freq[k], -1) * power_spec[k])
    #     spec_mom1 = spec_mom1 + (math.pow(freq[k], 1) * power_spec[k])
    #     spec_mom2 = spec_mom2 + (math.pow(freq[k], 2) * power_spec[k])
    #     spec_mom5 = spec_mom5 + (math.pow(freq[k], 5) * power_spec[k])
    #
    # f1 = spec_mom1 / spec_mom0
    # f2 = spec_mom0 / spec_mom2
    # f5 = spec_mom0 / spec_mom5

    powsum = 0
    powarray = []

    for l in range(len(freq)):
        # powsum = powsum + power_spec[l]
        powsum = integrate.simps(power_spec[:l+1],freq[:l+1])
        powarray.append(powsum)
        # print('yes')
        # freqarray.append(freq(l))

    mednum = powsum / 2

    meansumcombo = 0
    meansumpow = 0

    for p in range(len(freq)):
        meansumcombo = meansumcombo + (freq[p] * power_spec[p])
        meansumpow = meansumpow + (power_spec[p])

    # mean = meansumcombo / meansumpow
    print(meansumcombo)
    mean = meansumcombo / (sum(power_spec))
    print(mean)
    # analytic_signal = hilbert(emg1filt)
    # instant_phase = np.unwrap(np.angle(analytic_signal))
    # instant_freq = (np.diff(instant_phase)/(2.0*np.pi)*samplerate)
    # fig = plt.figure(5)
    # ax1 = fig.add_subplot(111,label = '1')
    # ax2 = fig.add_subplot(111,label = '2', frame_on = False)
    # ax1.plot(hor[1:],instant_freq)
    # ax2.plot(hor[1:],emg1filt[1:],color = 'green')
    #
    # plt.figure(6)
    # plt.plot(hor, analytic_signal)
    # plt.show()
    # exit()
    print(len(freq))
    print(freq[113])
    print(freq[226])
    exit()
    total_int = integrate.simps(power_spec[113:],freq[113:])
    print(total_int)
    # exit()
    for u in powarray:
        if u > (0.5*total_int):
            median = freq[powarray.index(u)-1]
            break

    print(median)
    # exit()
    # print(f1)
    print(mean)
    exit()
#     print(f2)
#     print(f5)
# # plt.show()


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

# from tkinter import *
# from tkinter import filedialog
# # from tkMessageBox import *
# import tkinter
#
# window = Tk()
# menu = Menu(window)
# d= {}
# # for i in enumerate(range(1,17)):
# #     d['v{}'.format(i)] = tkinter.IntVar()
# for i in range(1,17):
#     d['v{}'.format(i)] = tkinter.IntVar()
# # for i in range(1,17):
# #     print(d['v{}'.format(i)])
# # exit()
# # v[1] = tkinter.IntVar()
# # v[2] = tkinter.IntVar()
# # v[3] = tkinter.IntVar()
# # v[4] = tkinter.IntVar()
# # v[5] = tkinter.IntVar()
# # v[6] = tkinter.IntVar()
# # v[7] = tkinter.IntVar()
# # v[8] = tkinter.IntVar()
# # v[9] = tkinter.IntVar()
# # v[10] = tkinter.IntVar()
# # v[11] = tkinter.IntVar()
# # v[12] = tkinter.IntVar()
# # v[13] = tkinter.IntVar()
# # v[14] = tkinter.IntVar()
# # v[15] = tkinter.IntVar()
# # v[16] = tkinter.IntVar()
#
#
#
# emg_options = [('EMG 1',1),('EMG 2',2),('EMG 3',3),('EMG 4',4),
#                ('EMG 5',5),('EMG 6',6),('EMG 7',7),('EMG 8',8),
#                ('EMG 9',9),('EMG 10',10),('EMG 11',11),('EMG 12',12),
#                ('EMG 13',13),('EMG 14',14),('EMG 15',15),('EMG 16',16)]
#
# # file = filedialog.askopenfilename()
# filename = ''
# window.title('EMGProc')
# # window.geometry('800x600')
# csv_checker = 0
# color_iter = 0
# filenamesplit =[]
# emg_choice = []
# overwrite_choice = 0
# relative_height = 0.8
# pulse_width_min = 500
# pulse_width_max = 2000
# k_val = 6
# # import EMG_Proc_Func.py
#
#
# def change_color():
#     global color_iter
#     global bg_color
#     current_color = file_txt.cget("background")
#     next_color = bg_color if current_color == "red" else "red"
#     file_txt.config(background=next_color)
#     if color_iter < 4:
#         window.after(500, change_color)
#         color_iter += 1
#     else:
#         color_iter = 0
#         return
#
# def dataproc():
#     global csv_checker
#     global filename
#     global filenamesplit
#     global emg_choice
#     global overwrite_choice
#     global output_filename_choice
#     global relative_height
#     global pulse_width_min
#     global pulse_width_max
#     global k_val
#     # print(len(filename))
#     # print(filename)
#     # exit()
#
#     if csv_checker == 0:
#         from scipy import fftpack, signal
#         from matplotlib import pyplot as plt
#         import math
#         import numpy as np
#         import pandas as pd
#         import csv
#
#         import scipy.signal as sps
#         from scipy.signal import hilbert
#
#         from math import floor, log
#
#         # temp = []
#         samplerate = 1500
#         nyq = samplerate * 0.5
#         row_buff_val2 = 15
#         kmin = 2
#
#         def _linear_regression(x, y):
#             """Fast linear regression using Numba.
#             Parameters
#             ----------
#             x, y : ndarray, shape (n_times,)
#                 Variables
#             Returns
#             -------
#             slope : float
#                 Slope of 1D least-square regression.
#             intercept : float
#                 Intercept
#             """
#             n_times = x.size
#             sx2 = 0
#             sx = 0
#             sy = 0
#             sxy = 0
#             for j in range(n_times):
#                 sx2 += x[j] ** 2
#                 sx += x[j]
#                 sxy += x[j] * y[j]
#                 sy += y[j]
#             den = n_times * sx2 - (sx ** 2)
#             num = n_times * sxy - sx * sy
#             slope = num / den
#             intercept = np.mean(y) - slope * np.mean(x)
#             return slope, intercept
#
#         def _higuchi_fd(x, kmax):
#             n_times = x.size
#             lk = np.empty(kmax)
#             x_reg = np.empty(kmax)
#             y_reg = np.empty(kmax)
#             for k in range(1, kmax + 1):
#                 lm = np.empty((k,))
#                 for m in range(k):
#                     ll = 0
#                     n_max = floor((n_times - m - 1) / k)
#                     n_max = int(n_max)
#                     for j in range(1, n_max):
#                         ll += abs(x[m + j * k] - x[m + (j - 1) * k])
#                     ll /= k
#                     ll *= (n_times - 1) / (k * n_max)
#                     lm[m] = ll
#                 # Mean of lm
#                 m_lm = 0
#                 for m in range(k):
#                     m_lm += lm[m]
#                 m_lm /= k
#                 lk[k - 1] = m_lm
#                 x_reg[k - 1] = log(1. / k)
#                 y_reg[k - 1] = log(m_lm)
#             higuchi, _ = _linear_regression(x_reg, y_reg)
#             return higuchi
#
#         from openpyxl import Workbook
#         from openpyxl.chart import (ScatterChart, Reference, Series)
#         import openpyxl
#
#         # dest_filename = 'Subj08.xlsx'
#         if len(output_filename_choice.get())==0:
#             dest_filename = filename +'.xlsx'
#         else:
#             dest_filename = output_filename_choice.get()+'.xlsx'
#
#         if overwrite_choice == 0:
#             book = Workbook()
#         else:
#             book = openpyxl.load_workbook(dest_filename)
#         sheet = book.active
#         sheet.title = 'Data'
#
#         sheet2 = book.create_sheet()
#         sheet2.title = 'Graphs'
#         for h in range(len(filename)):
#             temp = []
#             with open(filename[h], 'r') as csvfile:
#                 csvreader = csv.reader(csvfile, delimiter=',')
#                 for row2 in csvreader:
#                     if csvreader.line_num == 3:
#                         temp.append(row2)
#                     if csvreader.line_num >= 6:
#                         if row2:
#                             temp.append(row2)
#                         else:
#                             break
#
#             df = pd.DataFrame(temp)  # turns the array into a dataframe
#             df.columns = df.iloc[0]  # sets the column names as the first row
#             df = df.drop(0)  # drops the first row since it is now a duplicate of the column names
#             df.reindex(df.index.drop(1))
#             df.reset_index(drop=True, inplace=True)
#             df.columns = ['frames', 'subframes', 'blank', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8',
#                       'emg9', 'emg10', 'emg11', 'emg12', 'emg13', 'emg14', 'emg15', 'emg16', 'unused7', 'unused8',
#                       'blank2']
#             df2 = df.drop(['frames', 'subframes', 'blank', 'unused7', 'unused8', 'blank2'], axis=1)
#             df2 = df2.astype(np.float)
#             print(len(df))
#             hor = np.arange(0, (len(df) - 0.5) / samplerate, 1 / samplerate)  # getting the time domain in seconds
#
#             emg = []
#             for i in range(0,len(emg_choice)):
#                 j = emg_choice[i]
#                 print(j)
#                 emg.append(df2['emg{}'.format(j)])
#
#             cutoff_freq = 20  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
#             cut = cutoff_freq / nyq
#             b, a = signal.butter(5, cut, btype='highpass', analog=False)
#             emg_high = []
#             for i in range(0, len(emg)):
#                 temp = signal.filtfilt(b, a, emg[i])
#                 emg_high.append(temp)
#             # print(emg_high[0])
#             # print(len(emg_high))
#             # exit()
#
#             cutoff_freq = 400  # ~500 Hz according to the emg book
#             cut = cutoff_freq / nyq
#             b, a = signal.butter(5, cut, btype='lowpass', analog=False)
#             emg_filt = []
#             for i in range(0, len(emg_high)):
#                 temp = signal.filtfilt(b, a, emg_high[i])
#                 emg_filt.append(temp)
#
#
#             lowcut = 10
#             highcut = 400
#             row_buff_val = 2 * (k_val-(kmin-1))
#             for t in range(len(emg_filt)):
#                 col = 1
#                 row2 = 2 + t*row_buff_val
#                 row = 1 + t*row_buff_val
#                 sheet.cell(row=row, column=col).value = 'EMG ' + str(emg_choice[t])
#                 row = row + 1
#                 # sheet.cell(row=row, column=col).value = 1
#                 cellcheck = 0
#                 samp_len = len(emg_filt[t])
#                 # q1_2 = floor(0.1 * samp_len)
#                 # overlap = floor(0.3 * q1_2)
#                 # print((len(emg_filt[t])-q1_2) // (overlap))
#                 # exit()
#                 # for k_iter in range(kmin,(k_val+1)):
#                 fractemp = []
#                 row = row + 1
#                 col = 1
#                 sheet.cell(row=row, column=col).value = 'Mean Value: ' + str(k_iter)
#
#                     for v in range((len(emg_filt[t])-q1_2) // overlap):
#
#                         emg_sec = q2[(0 + (overlap * v)):(q1_2 + (overlap * v))]
#                         fractemp.append(_higuchi_fd(emg_sec, floor(k_iter)))
#                     while cellcheck == 0:
#                         if sheet.cell(row=row, column=col).value != None:
#                             col = col + 1
#                         else:
#                             cellcheck = 1
#                     frac1 = np.average(fractemp)
#                     sheet.cell(row=row2, column=col).value = (col-1)
#                     sheet.cell(row= row, column=col).value = frac1
#                     col = col + 1
#
#                     cellcheck = 0
#
#
#             print(dest_filename)
#             book.save(filename=dest_filename)
#
#             gen_txt.configure(text='Generation Complete!', background = 'green')
#
#     else:
#         change_color()
#
#
# def filebrowse():
#     global csv_checker
#     global filename
#     global filenamesplit
#     gen_txt.configure(text='', background=bg_color)
#     output_filename_choice.config(state='normal')
#     kval.config(state='normal')
#
#     filename = filedialog.askopenfilenames(filetypes = (("Comma Separated Values","*.csv"),("all files","*.*")))
#     for w in range(len(filename)):
#         if filename[w].endswith('.csv'):
#             csv_checker = 0
#
#         else:
#             file_txt.configure(text='Please choose a csv file!')
#             csv_checker = 1
#             break
#
#     if csv_checker == 0:
#         filenamesplitfirst = filename[0].split('/')
#         filenamesplitlast = filename[-1].split('/')
#         filenamesplit1 = filenamesplitfirst[-1].split('_')
#         filenamesplit2 = filenamesplitlast[-1].split('_')
#         trial_numberfirst = filenamesplit1[-1].split('.')
#         trial_numberlast = filenamesplit2[-1].split('.')
#         file_txt.configure(text='Chosen Subject: ' + filenamesplit2[1] + '\nChosen TR #s: ' + trial_numberfirst[0] + ' - ' + trial_numberlast[0])
#         file_txt.config(background=bg_color)
#
# def get_emg_vals():
#     global emg_choice
#     # print(emg_choice)
#     emg_choice=[]
#     for i in range(1,17):
#         # print(i)
#         # print(d['v{}'.format(i)].get())
#         if d['v{}'.format(i)].get() == 1:
#             emg_choice.append(i)
#     print(emg_choice)
#
# def NewFile():
#     global emg_choice
#     global CB1
#     # global file_txt
#     file_txt.configure(text='Chosen File: ')
#     for i in range(1,17):
#         d['v{}'.format(i)].set(0)
#     gen_txt.configure(text='', background=bg_color)
#
# def NewFileKeyboard(self):
#     global emg_choice
#     global CB1
#     # global file_txt
#     file_txt.configure(text='Chosen File: ')
#     for i in range(1, 17):
#         d['v{}'.format(i)].set(0)
#     gen_txt.configure(text='', background=bg_color)
#
# def ExitKeyboard(self):
#     window.quit()
#
#
# def Tutorial():
#     top = Toplevel()
#     top.title("Tutorial")
#     # top.geometry('500x400')
#     msg1 = Label(top, text = 'Hello!'
#                             '\n Welcome to EMGProc!'
#                             '\nThis software was designed to work with a 16 count EMG system, as well as'
#                             '\n the csv file format created by Vicon Nexus (specifically v2.5 but possibly could'
#                             '\n work for other versions)'
#                             '\n'
#                             '\n'
#                             '\n How to use EMGProc:')
#     msg1.grid(row=1,column = 1,pady = (20,0),padx = 30)
#
#     msg2 = Label(top, text ='Step 1: Choose the csv file needed to process'
#                             '\nStep 2: Choose the output filename, as well as if a new output file should be created'
#                             '\n(overwriting any file with that name) or add on to an existing file created by EMGProc'
#                             '\nStep 3: Choose what EMGs were active and save them'
#                             '\nStep 4: Click Generate Output'
#                             '\nThe output file will be placed in the same location as the chosen file, unless you choose,'
#                             '\nyour own filename, in which case it will be placed in the location of this program.'
#                             '\n', justify = LEFT)
#     msg2.grid(row=2,column = 1,padx = 30)
#     msg3 = Label(top, text ='Keyboard Shortucts:')
#     msg3.grid(row=3,column = 1,padx = 30)
#     msg4 = Label(top, text='Ctrl+Q: New File'
#                             '\nCtrl+X: Exit', justify = LEFT)
#     msg4.grid(row=4,column = 1,pady = (0,20))
#
#     button1 = Button(top, text = 'Dismiss', command = top.destroy)
#     button1.grid(row=5,column = 1,pady = (0,20))
#
# def About():
#     top = Toplevel()
#     top.title("About")
#     # top.geometry('500x400')
#     msg1 = Label(top, text = 'EMGProc was created by Adam Lewis as part of a Master\'s Thesis Project'
#                              '\nCreated with PyCharm IDE and Python 3.6'
#                              '\nContact email: AdamLew73@gmail.com'
#                              '\n'
#                              '\nPlease do not copy or use without permission')
#     msg1.grid(row=1,column = 1,pady = (20,0),padx = 30)
#
#
#     button1 = Button(top, text='Dismiss', command=top.destroy)
#     button1.grid(row=2, column=1, pady=(0, 20))
#
# def focus_out_entry(event):
#     output_filename.config(text='\nOutput Filename: ' + output_filename_choice.get()+'.xlsx')
# def output_filename_save():
#     output_filename.config(text='\nOutput Filename: ' + output_filename_choice.get() + '.xlsx')
#
# def save_choice1():
#     global overwrite_choice
#     overwrite_choice = 0
#     print(overwrite_choice)
# def save_choice2():
#     global overwrite_choice
#     overwrite_choice = 1
#     print(overwrite_choice)
# def Pulse_Prop():
#     global k_val
#     k_val = int(kval.get())
#
#
# ##### GUI CREATION #####
# window.config(menu = menu)
#
# introtext = Label(window, text = '---------------------------------------------------------------------------------'
#                                '\nFile Management')
# introtext.grid(row=1,column = 2, columnspan = 3)
#
# btn1 = Button(window, text = 'Browse for File (csv only)', command = filebrowse)
# # btn1.pack(fill=X,padx = 100, pady = 50)
# btn1.grid(row=2,column = 3,pady = 20)
#
# file_txt = Label(window, text = 'Chosen File: ' + filename)
# # file_txt.pack(fill=X)
# file_txt.grid(row=3,column = 2, columnspan = 3, ipadx = 100)
# bg_color = file_txt.cget("background")
# out_text = ''
# output_filename_txt = Label(window, text = 'Desired Output Filename:')
# output_filename_txt.grid(row=4,column = 2, columnspan =2,padx = (25,100), sticky = W)
# output_filename_choice = Entry(window, textvariable = out_text, state = 'disabled', width = 50)
# output_filename_choice.bind("<Return>",focus_out_entry)
# output_filename_choice.bind("<FocusOut>",focus_out_entry)
# output_filename_choice.grid(row=4,column = 2, columnspan = 2, sticky = W, padx = (175,0))
# output_file_save = Button(window, text = 'Enter', command = output_filename_save)
# output_file_save.grid(row=4,column = 4, padx = (10,0), sticky = W)
# filesavechoicevar = IntVar()
# file_save_choice1 = Radiobutton(window, text = 'Overwrite File', variable = filesavechoicevar, value = 0, command = save_choice1)
# file_save_choice1.grid(row=3,column = 4, padx = (50,25), pady = (10,30), rowspan = 3, sticky = NE)
# file_save_choice2 = Radiobutton(window, text = 'Add to File', variable = filesavechoicevar, value = 1, command = save_choice2)
# file_save_choice2.grid(row=3,column = 4, padx = (0,40), pady = (35,0), rowspan = 3, sticky = NE)
#
# output_filename = Label(window, text = '\nOutput Filename: ')
# output_filename.grid(row=5,column = 3)
# septext = Label(window, text = '---------------------------------------------------------------------------------'
#                                '\nActive EMGs'
#                                '\n')
# septext.grid(row=6,column = 2, columnspan = 3)
#
# CB1 =Checkbutton(window, text = 'EMG 1', variable = d['v1'])
# CB1.grid(row=7, column = 2, ipadx = 50)
# CB2 =Checkbutton(window, text = 'EMG 2', variable = d['v2'])
# CB2.grid(row=8, column = 2)
# CB3 =Checkbutton(window, text = 'EMG 3', variable = d['v3'])
# CB3.grid(row=9, column = 2)
# CB4 =Checkbutton(window, text = 'EMG 4', variable = d['v4'])
# CB4.grid(row=10, column = 2)
# CB5 =Checkbutton(window, text = 'EMG 5', variable = d['v5'])
# CB5.grid(row=11, column = 2)
# CB6 =Checkbutton(window, text = 'EMG 6', variable = d['v6'])
# CB6.grid(row=12, column = 2)
# CB7 =Checkbutton(window, text = 'EMG 7', variable = d['v7'])
# CB7.grid(row=7, column = 3)
# CB8 =Checkbutton(window, text = 'EMG 8', variable = d['v8'])
# CB8.grid(row=8, column = 3)
# CB9 =Checkbutton(window, text = 'EMG 9', variable = d['v9'])
# CB9.grid(row=9, column = 3)
# CB10 =Checkbutton(window, text = 'EMG 10', variable = d['v10'])
# CB10.grid(row=10, column = 3)
# CB11 =Checkbutton(window, text = 'EMG 11', variable = d['v11'])
# CB11.grid(row=11, column = 3)
# CB12 =Checkbutton(window, text = 'EMG 12', variable = d['v12'])
# CB12.grid(row=7, column = 4, ipadx = 50)
# CB13 =Checkbutton(window, text = 'EMG 13', variable = d['v13'])
# CB13.grid(row=8, column = 4)
# CB14 =Checkbutton(window, text = 'EMG 14', variable = d['v14'])
# CB14.grid(row=9, column = 4)
# CB15 =Checkbutton(window, text = 'EMG 15', variable = d['v15'])
# CB15.grid(row=10, column = 4)
# CB16 =Checkbutton(window, text = 'EMG 16', variable = d['v16'])
# CB16.grid(row=11, column = 4)
#
# btn2 = Button(window, text = 'Save Active EMGs', command = get_emg_vals)
# btn2.grid(row=13,column = 3,pady = 10)
# # file_txt = Label(window, text = 'Chosen File: ' + filename)
# # file_txt.pack(fill=X)
#
# septext2 = Label(window, text = '---------------------------------------------------------------------------------'
#                                '\nKMax Definitions'
#                                '\n')
# septext2.grid(row=14,column = 2, columnspan = 3)
#
#
# v2 = StringVar(window, value='6')
#
#
# kval_Text = Label(window, text = 'K_Max Value')
# kval_Text.grid(row=15,column = 2, columnspan = 3, padx = (50,50))
# kval = Entry(window, state = 'disabled', textvariable = v2, width = 10)
# kval.grid(row=16,column = 2, columnspan = 3, padx = (45,50))
#
# Pulse_Button = Button(window, text = 'Save Properties', command = Pulse_Prop)
# Pulse_Button.grid(row=17,column = 3,pady = 20)
#
# septext3 = Label(window, text = '---------------------------------------------------------------------------------')
# septext3.grid(row=18,column = 2, columnspan = 3)
#
# btn3 = Button(window, text = 'Generate Output', command = dataproc)
# btn3.grid(row=19,column = 3)
# gen_txt = Label(window, text = '')
# gen_txt.grid(row=20,column = 3, ipadx = 150, pady = (10,20))
# # gen_txt.grid(row=20,column = 2,columnspan = 3, ipadx = 150, pady = (10,20))
#
# # btn2.grid(fill=X,padx = 100, pady = 220)
#
#
# filemenu = Menu(menu, tearoff = False)
# filemenu2 = Menu(menu, tearoff = False)
# filemenu3 = Menu(menu, tearoff = False)
#
# menu.add_cascade(label='File', menu = filemenu)
# filemenu.add_command(label='New', command=NewFile, accelerator ="Ctrl+Q")
# window.bind_all("<Control-q>",NewFileKeyboard)
# filemenu.add_separator()
# filemenu.add_command(label='Exit', command=window.quit)
# window.bind_all("<Control-x>",ExitKeyboard)
# menu.add_cascade(label='Help', menu = filemenu2)
# filemenu2.add_command(label='Tutorial', command=Tutorial)
#
# menu.add_cascade(label='Info', menu = filemenu3)
# filemenu3.add_command(label='About', command=About)
#
#
# window.mainloop()