### CSV Extraction Code with help from: Jim Todd of Stack Overflow
from scipy import fftpack, signal
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
import scipy.signal as sps
from scipy.signal import hilbert
temp = []
samplerate = 1500
nyq = samplerate*0.5
# np.set_printoptions(threshold=np.inf)
def test_func(x,a,b):
    return a*np.sin(b*x)

# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj01/Free/RF_Subj01_Free_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj01/Free/RF_Subj01_Free_Fatigue5_TR28.csv', 'r') as csvfile:


# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj04/Free/RF_Subj04_Free_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj01/Free/RF_Subj04_Free_Fatigue5_TR25.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj08/Free/RF_Subj08_Free_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj08/Free/RF_Subj08_Free_Fatigue5_TR24.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj08/Hoka/RF_Subj08_Hoka_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj08/Hoka/RF_Subj08_Hoka_Fatigue5_TR23.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj07/Free/RF_Subj07_Free_Fatigue5_TR02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj07/Free/RF_Subj07_Free_Fatigue5_TR25.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj07/Hoka/RF_Subj07_Hoka_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj07/Hoka/RF_Subj07_Hoka_Fatigue5_TR21.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj06/Free/RF_Subj06_Free_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj06/Free/RF_Subj06_Free_Fatigue5_TR23.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj06/Hoka/RF_Subj06_Hoka_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj06/Hoka/RF_Subj06_Hoka_Fatigue5_TR22.csv', 'r') as csvfile:




# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg3/Biodex_Leg_Test03.csv', 'r') as csvfile:
# with open('C:/Users/sword/PycharmProjects/excelimport/PT01/Free/RF2_SubjP01_Free_Biodex_TR01_Fixed.csv', 'r') as csvfile:
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
df.columns = ['frames', 'subframes', 'blank', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8', 'emg9', 'emg10', 'emg11', 'emg12', 'emg13', 'emg14', 'emg15', 'emg16', 'unused7', 'unused8', 'blank2']
# df.columns = ['frames', 'subframes', 'blank', 'NU1', 'NU2', 'NatTA', 'NU3', 'NatBic', 'NatTri', 'NatGastLat', 'NatGastMed', 'EmilyBic', 'EmilyTri', 'EmilyGastMed', 'EmilyGastLat', 'EmilyTA', 'AdamBic', 'AdamTri', 'unused6', 'unused7', 'unused8', 'blank2']
# df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused4', 'unused5', 'unused6', 'unused7', 'unused8', 'blank2']
# df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused7', 'unused8', 'blank2']
df2 = df.drop(['frames', 'subframes', 'blank', 'unused7', 'unused8', 'blank2'], axis = 1)
#print(df2)
df2 = df2.astype(np.float)
print(len(df))
hor = np.arange(0, (len(df)-0.5)/samplerate, 1/samplerate)  #getting the time domain in seconds
emg1 = df2.emg8
emg3 = df2.emg9
emg5 = df2.emg11
emg7 = df2.emg1
emg9 = df2.emg1
print(len(hor))
#
# emgtemp = emg1[96000:837000] ##For adam running trial (biodex second 3)
# emg1 = []

# emgtemp = emg1[84900:924000] ##For adam biodex3 trial 1
# emg1 = []

# emgtemp = emg1[79500:1212000] ##For adam biodex3 trial 2
# emg1 = []


# emg1 = emg1[292500:658500]  ##For adam biodex right leg 1
# hor = hor[292500:658500]

# emg1 = emg1[96000:837000]  ##For adam running trial (biodex second 3)
# hor = hor[96000:837000]
#
# for i in emgtemp:
#     emg1.append(i)
# hor = np.arange(0, (len(emg1)-0.5)/samplerate, 1/samplerate)

# zero= []
# average = np.mean(emg1)
# for i in range(len(hor)):
#     zero.append(average)
# fig = plt.figure(8)
# ax0 = fig.add_subplot(1,1,1)
# ax0.plot(hor, zero, color = 'red')
# ax0.plot(hor, emg1)
# plt.title('Biceps')

# plt.show()
# exit()

emg2 = []
for i in range(len(emg1)):
    if emg1[i] < 0.005:
        emg2.append(emg1[i])
zero1= []
average = np.mean(emg2)
for i in range(len(hor)):
    zero1.append(average)

emg4 = []
for i in range(len(emg3)):
    if emg3[i] < 0.005:
        emg4.append(emg3[i])
zero3= []
average = np.mean(emg4)
for i in range(len(hor)):
    zero3.append(average)

emg6 = []
for i in range(len(emg5)):
    if emg5[i] < 0.005:
        emg6.append(emg5[i])
zero5= []
average = np.mean(emg6)
for i in range(len(hor)):
    zero5.append(average)

emg8 = []
for i in range(len(emg7)):
    if emg7[i] < 0.005:
        emg8.append(emg7[i])
zero7 = []
average = np.mean(emg8)
for i in range(len(hor)):
        zero7.append(average)

emg10 = []
for i in range(len(emg9)):
    if emg9[i] < 0.005:
        emg10.append(emg9[i])
zero9 = []
average = np.mean(emg10)
for i in range(len(hor)):
    zero9.append(average)


fig = plt.figure(1)
ax0 = fig.add_subplot(2,3,1)
ax0.plot(hor, zero1, color = 'red')
ax0.plot(hor, emg1)

ax0 = fig.add_subplot(2,3,2)
ax0.plot(hor, zero3, color = 'red')
ax0.plot(hor, emg3)

ax0 = fig.add_subplot(2,3,3)
ax0.plot(hor, zero5, color = 'red')
ax0.plot(hor, emg5)

ax0 = fig.add_subplot(2,3,4)
ax0.plot(hor, zero7, color = 'red')
ax0.plot(hor, emg7)

ax0 = fig.add_subplot(2,3,5)
ax0.plot(hor, zero9, color = 'red')
ax0.plot(hor, emg9)

# plt.show()
# exit()


cutoff_freq = 20  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='highpass', analog=False)
emg1high = signal.filtfilt(b, a, emg1)

cutoff_freq = 400  # ~500 Hz according to the emg book
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='lowpass', analog=False)
emg1filt = signal.filtfilt(b, a, emg1high)


# emg1filt = emg





## Auto Detect Musc On ##
emg_rec = abs(emg1filt)
# end = len(emg_rec)-1
slopes = []
# akima = []
#
# for i in range(0,end):
#     # slopes[i]= (emg_rec[i+1]-emg_rec[i])/(hor[i+1]-hor[i])
#     slopes.append((emg_rec[i + 1] - emg_rec[i]) / (hor[i + 1] - hor[i]))
# slop = np.array(slopes).transpose()
# print(slop)
# print(slopes)
# for i in range(2,len(slopes)-2):
#     akima.append(    ((abs(slopes[i+1]-slopes[i])*slopes[i-1])+(abs(slopes[i-1]-slopes[i-2])*slopes[i]))/abs(slopes[i+1]-slopes[i])+abs(slopes[i-1]-slopes[i-2])         )
# # print(akima)
# ynew = signal.savgol_filter(emg_rec,811,2)
ynew = signal.savgol_filter(emg_rec,1501,2)
# ynew = 2.2*ynew
# ynew2 = signal.savgol_filter(ynew, 51, 6)
cutoff_freq = 10  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='lowpass', analog=False)
ynew2 = signal.filtfilt(b, a, ynew)

ynew3 = signal.savgol_filter(ynew2,1501,2)


ymean = np.average(ynew)
yml = []
for i in hor:
    yml.append(ymean*0.9)

yml25 = []
for i in hor:
    yml25.append(ymean*0.5)

end = len(ynew3)-1
for i in range(0,end):
    # slopes[i]= (emg_rec[i+1]-emg_rec[i])/(hor[i+1]-hor[i])
    slopes.append((ynew3[i + 1] - ynew3[i]) / (hor[i + 1] - hor[i]))
slopes.append(0)  ##temp, delete later
slop = np.array(slopes).transpose()



checker=0
points = []
pointsT = []
last = 0
for i in range(0,len(slop)):
    # if ynew3[i]< yml25[0]:
    if slop[i] > 0:
        if last < 0:
            checker+=1
            points.append(ynew3[i-1])
            pointsT.append(hor[i-1])
    last = slop[i]
print(checker)
#######################
checker2 = 0
flag = 0
for i in range(0,len(slop)):
    if ynew3[i]< yml25[0]:
        if flag == 0:
            checker2+=1
            flag = 1
        elif flag == 1:
            continue
    elif ynew3[i]>yml25[0]:
        if flag == 0:
            continue
        elif flag == 1:
            flag = 0
print(checker2)

crossp = []
crossT = []
crossind = []

for i in range(0,len(hor)):
    if ynew3[i] > yml25[0]:
        crossp.append(emg_rec[i])
        crossT.append(hor[i])

x1 = np.linspace(0, len(crossp),len(crossp))

crossupper = []
crossTupper = []
crosslower = []
crossTlower = []
crossind2 = []
crossupperind = []
crosslowerind = []
temparraymaxchecker = []
firstpointflag = 0
tempwidth = 0
widtharray = []
firstwidthflag = 0
twopointsrecordedflag = 0
pointswitch = 0
discon_count = 0
discon_time_array = []
print(len(hor))
print(len(slop))
print('This is end of test')
for i in range(1,len(hor)):
    if (ynew3[i] > yml[0] and ynew3[i-1] < yml[0]):
        if pointswitch == 0:
            crossupper.append(yml[i-1])
            crossTupper.append(hor[i-1])
            crossupperind.append((i-1))
            pointswitch = 1

    # elif (ynew3[i] < yml25[0] and ynew3[i-1] > yml25[0]):
    # elif pointswitch == 1:
    #     if crossTlower:
    #         print(crossTlower[-1])
    #         print(crossTupper[-1])
    #
    #         if (crossTupper[-1] - crossTlower[-1]) < 0.3:
    #             del crossTupper[-1]
    #             del crossupper[-1]
    #             del crossupperind[-1]
    #             pointswitch = 0
    # elif (ynew3[i] < yml25[0] and ynew3[i - 1] > yml25[0]):
    elif (ynew3[i] < (yml25[0]) and ynew3[i - 1] > (yml25[0])):
        if pointswitch == 1:
            # if firstpointflag == 0:
            #     firstpointflag = 1
            #     continue
            crosslower.append(yml25[i-1])
            crossTlower.append(hor[i-1])
            crosslowerind.append((i-1))
            twopointsrecordedflag = 1
            pointswitch = 0
    # print(crossTupper)
    # print(crossTlower)
    elif (slop[i] > 0 and slop[i - 1] < 0 and ynew3[i] < yml[0]):
        if pointswitch == 1:
            # if firstpointflag == 0:
            #     firstpointflag = 1
            #     continue
            crosslower.append(yml25[i-1])
            crossTlower.append(hor[i-1])
            crosslowerind.append((i-1))
            twopointsrecordedflag = 1
            pointswitch = 0
    # print(crossTupper)
    # print(crossTlower)
    if (crossupper is not None and crosslower is not None):
        deleteflag = 0
        if twopointsrecordedflag == 1:
            if firstwidthflag == 0:
                tempwidth = crossTlower[0] - crossTupper[0]
                # tempwidth = 1.41
                widtharray.append(tempwidth)
                firstwidthflag = 1
            else:
                tempwidth = crossTlower[-1] - crossTupper[-1]
                # print(tempwidth)
                if (tempwidth > (0.1* np.median(widtharray)) and tempwidth < (2 * np.median(widtharray))):
                    widtharray.append(tempwidth)
                else:
                    del crossTlower[-1]
                    del crosslower[-1]
                    del crossTupper[-1]
                    del crossupper[-1]
                    del crosslowerind[-1]
                    del crossupperind[-1]
                    deleteflag = 1
                if deleteflag == 0:
                    # print(crossupperind[-1])
                    # print(crosslowerind[-1])
                    temparraymaxchecker.extend(emg_rec[crossupperind[-1]:crosslowerind[-1]])
                    if np.amax(temparraymaxchecker)>0.005:
                        discon_count = discon_count + 1
                        discon_time_array.append(hor[crossupperind])
                        del crossTlower[-1]
                        del crosslower[-1]
                        del crossTupper[-1]
                        del crossupper[-1]
                        del crosslowerind[-1]
                        del crossupperind[-1]

                    temparraymaxchecker = []


            twopointsrecordedflag = 0
#### NOTE THE ABOVE SECTION IS MOSTLY IDENTICAL TO THE BELOW SECTION. THIS IS IN ORDER TO FIND THE MEDIAN WIDTH OF THE WHOLE TRIAL####

crossupper = []
crossTupper = []
crosslower = []
crossTlower = []
crossind2 = []
crossupperind = []
crosslowerind = []
firstpointflag = 0
twopointsrecordedflag = 0
pointswitch = 0

discon_count = 0
discon_time_array = []
temparraymaxchecker = []
for i in range(1,len(hor)):
    if (ynew3[i] > yml[0] and ynew3[i-1] < yml[0]):
        if pointswitch == 0:
            crossupper.append(yml[i-1])
            crossTupper.append(hor[i-1])
            crossupperind.append((i-1))
            pointswitch = 1
    elif (ynew3[i] < (yml25[0]) and ynew3[i - 1] > (yml25[0])):
        if pointswitch == 1:
            crosslower.append(yml25[i-1])
            crossTlower.append(hor[i-1])
            crosslowerind.append((i-1))
            twopointsrecordedflag = 1
            pointswitch = 0
    elif (slop[i] > 0 and slop[i - 1] < 0 and ynew3[i] < yml[0]):
        if pointswitch == 1:
            crosslower.append(yml25[i-1])
            crossTlower.append(hor[i-1])
            crosslowerind.append((i-1))
            twopointsrecordedflag = 1
            pointswitch = 0
    if (crossupper is not None and crosslower is not None):
        deleteflag = 0
        if twopointsrecordedflag == 1:
            tempwidth = crossTlower[-1] - crossTupper[-1]

            if (tempwidth < (0.5* np.median(widtharray)) or tempwidth > (1.5* np.median(widtharray))):   ######## using 1.5x is normal, used 3x for a slow set

                del crossTlower[-1]
                del crosslower[-1]
                del crossTupper[-1]
                del crossupper[-1]
                del crosslowerind[-1]
                del crossupperind[-1]
                deleteflag = 1

            if deleteflag == 0:
                # print(crossupperind[-1])
                # print(crosslowerind[-1])
                temparraymaxchecker.extend(emg_rec[crossupperind[-1]:crosslowerind[-1]])
                # if hor[crossupperind[-1]] > 200:
                #     print('power:' + str(np.amax(temparraymaxchecker)))
                if np.amax(temparraymaxchecker) > 0.005:
                    discon_count = discon_count + 1
                    discon_time_array.append(hor[crossupperind])
                    del crossTlower[-1]
                    del crosslower[-1]
                    del crossTupper[-1]
                    del crossupper[-1]
                    del crosslowerind[-1]
                    del crossupperind[-1]

                temparraymaxchecker = []



        twopointsrecordedflag = 0





print(discon_count)



print('End of Loop')
print(len(crossTupper))
print(len(crossTlower))
print(len(crossTlower)/2)
if len(crossTlower) != len(crossTupper):
    del crossTupper[-1]
    del crossupper[-1]


M = 0
# N = 45
N = len(crosslowerind)
sectionpoints = []
sectionT = []

for i in range(M,N):
    sectionpoints.extend(emg_rec[crossupperind[i]:crosslowerind[i]])
    sectionT.extend(hor[crossupperind[i]:crosslowerind[i]])

x2 = np.linspace(0, len(sectionpoints), len(sectionpoints))
x2 = x2/1500

# print('Start Time')
#
# for i in range(0, len(crossTupper)):
#     print(crossTupper[i])
#
# print('End Time')
# for i in range(0, len(crossTlower)):
#     print(crossTlower[i])

metronome = []
last1 = slop[0]
timecount = 0
first_slop_flag = 0
# for i in range(1,len(slop)):
#     if first_slop_flag == 0:
#         metronome.append(hor[0])
#         first_slop_flag = 1
#     elif first_slop_flag == 1:
#         timecount += 1
#         if timecount == 2145:
#             timecount = 0
#             metronome.append(hor[i])
#     last1 = slop[i]
# print("METRONOME")
# print(metronome)

fig = plt.figure(2)
ax0 = fig.add_subplot(1,1,1)
ax0.plot(hor, emg_rec)
# ampenv = hilbert(emg1rec)
# ax0.plot(hor, ynew, color = 'red')
# ax0.plot(hor, yml, color = 'khaki')
ax0.plot(crossT, crossp,color = 'green')
ax0.plot(hor, yml, color = 'khaki')
ax0.plot(hor, yml25, color = 'purple')
# ax0.plot(hor, ynew2, color = 'purple')

ax0.plot(hor, ynew3, color = 'purple')
ax0.plot(pointsT, points, marker = "x",color = 'red', linestyle = "None")
# ax0.plot(crossT, crossp, marker = "*",color = 'magenta', linestyle = "None")
# ax0.plot(crossT, crossp,color = 'green')

# ax0.plot(keepme1T, keepme1,color = 'green')
# ax0.plot(temt1, tem1, marker = "x", color = 'purple', linestyle = "None")
# ax0.plot(crosskeepT, crosskeep,color = 'green')

# ax0.plot(crossT[600], 0.0005, color = 'magenta', marker = "x")
ax0.plot(crossTupper, crossupper, marker = "x",color = 'yellow', linestyle = "None")
ax0.plot(crossTlower, crosslower, marker = "x",color = 'yellow', linestyle = "None")
# ax0.plot(risingT, rising, marker = 'o', color = 'purple')
# fig = plt.figure(3)
# ax0 = fig.add_subplot(1,1,1)
# ax0.plot(hor, emg_rec)
# ax0.plot(hor, ymean, color = 'red')
# for i in range(0, len(metronome)):
#     ax0.axvline(metronome[i], color = 'purple')
# ax0.plot(hor[2500:3500], yml[2500:3500], color = 'khaki')




#
# fig = plt.figure(3)
# ax0 = fig.add_subplot(1,1,1)
# ax0.plot(x1,crossp)

fig = plt.figure(3)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(sectionT, sectionpoints)

fig = plt.figure(4)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(x2, sectionpoints)

sectionpoints_array = np.asarray(sectionpoints)
plt.figure(5)
f1,t1, Sxx1 = sps.spectrogram(sectionpoints_array,samplerate,('hamming'),128,0, scaling= 'density')
plt.pcolormesh(t1,f1,Sxx1)
plt.ylim((0,300))
plt.title('Biceps')

###ISO POWER###
# print(Sxx1.shape)
# print(len(Sxx1))
isopowavg = np.average(Sxx1[:20,:], axis = 1)
# print(isopowavg.shape)
# for i in range(0,20):
#     print(isopowavg[i])

# exit()
###################

plt.figure(6)
plt.specgram(sectionpoints_array,Fs=samplerate,NFFT=128,noverlap=0)

freq1, power_spec1 = signal.periodogram(sectionpoints_array, samplerate)

meantest1 = sum(freq1*power_spec1)/sum(power_spec1)
# print('Mean frequency is: ' + str(round(meantest1, 2))+' Hz')

plt.figure(7)
plt.plot(hor, emg_rec)

plt.show()
