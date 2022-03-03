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
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR17.csv', 'r') as csvfile:     #Emily Calf Raise Dyn Set 1 ########
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR18.csv', 'r') as csvfile:     #Nathan Calf Raise Dyn Set 2
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR19.csv', 'r') as csvfile:     #Emily Calf Raise Dyn Set 2

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/6-10-2019 Adam3D and MR3 Proc/Adam3D Test Data/New Session 1/Adam_3D_Iso_TR01.csv', 'r') as csvfile:
with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/6-10-2019 Adam3D and MR3 Proc/Adam3D Test Data/New Session 1/Adam_3D_Iso_TR02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/6-10-2019 Adam3D and MR3 Proc/Adam3D Test Data/New Session 1/Adam_3D_Iso_TR03.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/6-10-2019 Adam3D and MR3 Proc/Adam3D Test Data/New Session 1/Adam_3D_Iso_TR04.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/6-10-2019 Adam3D and MR3 Proc/Adam3D Test Data/New Session 1/Adam_3D_Iso_TR05.csv', 'r') as csvfile:

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
emg1 = df2.NatTA
print(len(hor))
plt.figure(1)
plt.plot(hor, emg1)
plt.title('Biceps')

cutoff_freq = 5  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='highpass', analog=False)
emg1high = signal.filtfilt(b, a, emg1)

cutoff_freq = 400  # ~500 Hz according to the emg book
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='lowpass', analog=False)
emg1filt = signal.filtfilt(b, a, emg1high)

## Auto Detect Musc On ##
emg_rec = abs(emg1filt)
# end = len(emg_rec)-1
slopes = []


#
# ynew3 = signal.savgol_filter(emg1filt,255,2)

yenv1 = hilbert(emg1filt)
yenv = np.abs(yenv1)

ynew3 = signal.savgol_filter(yenv,2501,2)


fig = plt.figure(2)
ax0 = fig.add_subplot(1,1,1)
ax0.plot(hor, emg1filt)

ax0.plot(hor, ynew3, color = 'purple')
ax0.plot(hor, yenv, color = 'red')
ax0.plot(hor, ynew3, color = 'green')
plt.show()



ymean = np.average(yenv)
yml = []
for i in hor:
    yml.append(ymean)

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
firstpointflag = 0
tempwidth = 0
widtharray = []
firstwidthflag = 0
twopointsrecordedflag = 0
pointswitch = 0
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
        if twopointsrecordedflag == 1:
            if firstwidthflag == 0:
                # tempwidth = crossTlower[0] - crossTupper[0]
                tempwidth = 1.41
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
        if twopointsrecordedflag == 1:
            tempwidth = crossTlower[-1] - crossTupper[-1]
            if (tempwidth < (0.5* np.median(widtharray)) or tempwidth > (1.5* np.median(widtharray))):

                del crossTlower[-1]
                del crosslower[-1]
                del crossTupper[-1]
                del crossupper[-1]
                del crosslowerind[-1]
                del crossupperind[-1]
        twopointsrecordedflag = 0


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

print('Start Time')

for i in range(0, len(crossTupper)):
    print(crossTupper[i])

print('End Time')
for i in range(0, len(crossTlower)):
    print(crossTlower[i])

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

# # fig = plt.figure(2)
# # ax0 = fig.add_subplot(1,1,1)
# # ax0.plot(hor, emg1filt)
#
# # ampenv = hilbert(emg1rec)
# # ax0.plot(hor, ynew, color = 'red')
# # ax0.plot(hor, yml, color = 'khaki')
# # ax0.plot(crossT, crossp,color = 'green')
# # ax0.plot(hor, yml, color = 'khaki')
# # ax0.plot(hor, yml25, color = 'purple')
# # ax0.plot(hor, ynew2, color = 'purple')
#
# ax0.plot(hor, ynew3, color = 'purple')
# ax0.plot(pointsT, points, marker = "x",color = 'red', linestyle = "None")
# ax0.plot(crossT, crossp, marker = "*",color = 'magenta', linestyle = "None")
# ax0.plot(crossT, crossp,color = 'green')

# ax0.plot(keepme1T, keepme1,color = 'green')
# ax0.plot(temt1, tem1, marker = "x", color = 'purple', linestyle = "None")
# ax0.plot(crosskeepT, crosskeep,color = 'green')

# ax0.plot(crossT[600], 0.0005, color = 'magenta', marker = "x")
# ax0.plot(crossTupper, crossupper, marker = "x",color = 'yellow', linestyle = "None")
# ax0.plot(crossTlower, crosslower, marker = "x",color = 'yellow', linestyle = "None")
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

# fig = plt.figure(3)
# ax0 = fig.add_subplot(1, 1, 1)
# ax0.plot(sectionT, sectionpoints)
#
# fig = plt.figure(4)
# ax0 = fig.add_subplot(1, 1, 1)
# ax0.plot(x2, sectionpoints)

# sectionpoints_array = np.asarray(sectionpoints)
# plt.figure(5)
# f1,t1, Sxx1 = sps.spectrogram(sectionpoints_array,samplerate,('hamming'),128,0, scaling= 'density')
# plt.pcolormesh(t1,f1,Sxx1)
# plt.ylim((0,300))
# plt.title('Biceps')
#
#
# plt.figure(6)
# plt.specgram(sectionpoints_array,Fs=samplerate,NFFT=128,noverlap=0)
#
# freq1, power_spec1 = signal.periodogram(sectionpoints_array, samplerate)
#
# meantest1 = sum(freq1*power_spec1)/sum(power_spec1)
# print('Mean frequency is: ' + str(round(meantest1, 2))+' Hz')

# plt.figure(7)
# plt.plot(hor, emg_rec)

plt.show()
