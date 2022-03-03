### CSV Extraction Code with help from: Jim Todd of Stack Overflow
from scipy import fftpack, signal
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
import math
import scipy.integrate as int
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

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Andrew_Arm/Dim_AnAd_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Andrew_Arm/Dim_AnAd_TR02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Andrew_Arm/Dim_AnAd_TR03.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Andrew_Arm/Dim_AnAd_TR04.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Andrew_Arm/Dim_AnAd_TR05.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Andrew_Arm/Dim_AnAd_TR06.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Andrew_Arm/Dim_AnAd_TR07.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex/Adam_Biodex_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex/Adam_Biodex_TR02.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg/Adam_Biodex_Leg01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg/Adam_Biodex_Leg02.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg2/Adam_Biodex_Leg_Second01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg2/Adam_Biodex_Leg_Second02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg2/Adam_Biodex_Leg_Second03.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg3/Biodex_Leg_Test01.csv', 'r') as csvfile:
with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg3/Biodex_Leg_Test02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg3/Biodex_Leg_Test03.csv', 'r') as csvfile:

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
emg1 = df2.emg11
print(len(hor))


# emgtemp = emg1[96000:837000] ##For adam running trial (biodex second 3)
# emg1 = []

# emgtemp = emg1[84900:924000] ##For adam biodex3 trial 1
# emg1 = []

emgtemp = emg1[79500:1212000] ##For adam biodex3 trial 2
emg1 = []


# emg1 = emg1[277500:513000]  ##For adam biodex right leg 2
# hor = hor[277500:513000]



for i in emgtemp:
    emg1.append(i)
hor = np.arange(0, (len(emg1)-0.5)/samplerate, 1/samplerate)


plt.figure(8)
plt.plot(hor, emg1)
plt.title('Biceps')


emg23 = []
for i in range(len(emg1)):
    if emg1[i] < 0.005:
        emg23.append(emg1[i])

# print('we did it bois')
hor2 = np.arange(0, (len(emg23)-0.5)/samplerate, 1/samplerate)  #getting the time domain in seconds
emg1 = emg23
hor = hor2

print(len(hor))
plt.figure(1)
plt.plot(hor, emg1)
plt.title('Biceps')
# plt.show()
# exit()

cutoff_freq = 10  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
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
ynew = signal.savgol_filter(emg_rec,1501,2)


cutoff_freq = 10  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='lowpass', analog=False)
ynew2 = signal.filtfilt(b, a, ynew)

ynew3 = signal.savgol_filter(ynew2,1501,2)

#######################################################################################################
ymean = np.average(ynew)
yml = []
for i in hor:
    yml.append(0.85*ymean)

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
            if (tempwidth < (0.5* np.median(widtharray)) or tempwidth > (1.5* np.median(widtharray))):  ######## using 1.5x is normal, used 3x for a slow set

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


# M = 50
# N = 51
# # N = len(crosslowerind)
# print('pulses: ' + str(len(crosslowerind)))
# sectionpoints = []
# sectionT = []
#
# for i in range(M,N):
#     sectionpoints.extend(emg_rec[crossupperind[i]:crosslowerind[i]])
#     sectionT.extend(hor[crossupperind[i]:crosslowerind[i]])
#
# x2 = np.linspace(0, len(sectionpoints), len(sectionpoints))
# x2 = x2/1500
#
#
# sectionpoints_array = np.asarray(sectionpoints)
#
# freq1, power_spec1 = signal.periodogram(sectionpoints_array, samplerate)
#
# lowcut = 10
# highcut = 400
#
# summ1 = 0
#
# # print(len(freq1))
# # print(freq1[663])
# test_num = math.pow(10,-1)
# print(test_num)
#
# spec_mom0 = 0
# spec_mom1 = 0
# spec_mom2 = 0
# spec_mom3 = 0
# spec_mom4 = 0
# spec_mom5 = 0
# #from 10-400 Hz
# for i in range(17,664):
#     spec_mom0 = spec_mom0 + (math.pow(freq1[i],-1)*power_spec1[i])
#     spec_mom1 = spec_mom1 + (math.pow(freq1[i], 1)*power_spec1[i])
#     spec_mom2 = spec_mom2 + (math.pow(freq1[i], 2)*power_spec1[i])
#     spec_mom3 = spec_mom3 + (math.pow(freq1[i], 3)*power_spec1[i])
#     spec_mom4 = spec_mom4 + (math.pow(freq1[i], 4)*power_spec1[i])
#     spec_mom5 = spec_mom5 + (math.pow(freq1[i], 5)*power_spec1[i])
#
# f2 = spec_mom0/spec_mom2
# f3 = spec_mom0/spec_mom3
# f4 = spec_mom0/spec_mom4
# f5 = spec_mom0/spec_mom5
# powsum = 0
# powarray = []
# for i in range(len(freq1)):
#     powsum = powsum + power_spec1[i]
#     powarray.append(powsum)
# mednum = powsum/2
# for i in powarray:
#     if i > mednum:
#         median = freq1[powarray.index(i)]
#         print('median: ' + str(median))
#         break
# #
# #
# # testmedian =
# # print(testmedian)
# # print(spec_mom0)
# # print('')
# print(f2)
# print(f3)
# print(f4)
# print(f5)

print('pulses: ' + str(len(crosslowerind)))
M = 0
N = 1
lowcut = 10
highcut = 400
for a in range((len(crosslowerind))):
# for a in range((len(crosslowerind)-2)): ##for arm biodex set 1
# for a in range((len(crosslowerind)-6)):   ##for arm biodex set 2
# for a in range(math.floor(len(crosslowerind)/2)):
# N = len(crosslowerind)

    sectionpoints = []
    sectionT = []
    lowhalfsec = []
    lowhalfT = []
    upperhalfsec = []
    upperhalfT = []
    for i in range(M,N):
        sectionpoints.extend(emg_rec[crossupperind[i]:crosslowerind[i]])
        sectionT.extend(hor[crossupperind[i]:crosslowerind[i]])
        lowhalfsec.extend(emg_rec[crossupperind[i]:(crossupperind[i] + math.floor((crosslowerind[i] - crossupperind[i]) / 2))])
        lowhalfT.extend(hor[crossupperind[i]:(crossupperind[i] + math.floor((crosslowerind[i] - crossupperind[i]) / 2))])
        upperhalfsec.extend(emg_rec[(crossupperind[i] + math.floor((crosslowerind[i] - crossupperind[i]) / 2)):crosslowerind[i]])
        upperhalfT.extend(hor[(crossupperind[i] + math.floor((crosslowerind[i] - crossupperind[i]) / 2)):crosslowerind[i]])
    x2 = np.linspace(0, len(sectionpoints), len(sectionpoints))
    x2 = x2/1500
    x3 = np.linspace(0, len(lowhalfsec), len(lowhalfsec))
    x3 = x3/1500
    x4 = np.linspace(0, len(upperhalfsec), len(upperhalfsec))
    x4 = x4/1500

    sectionpoints_array = np.asarray(sectionpoints)
    lowhalf_array = np.asarray(lowhalfsec)
    upperhalf_array = np.asarray(upperhalfsec)



    freq1, power_spec1 = signal.periodogram(sectionpoints_array, samplerate)
    freq2, power_spec2 = signal.periodogram(lowhalf_array, samplerate)
    freq3, power_spec3 = signal.periodogram(upperhalf_array, samplerate)

    summ1 = 0
    summ2 = 0
    #
    # print(len(freq1))
    # print(freq1[17])
    # print(freq1[629])

    for c in range(len(freq1)):
        if abs(freq1[c]-10) < 1:
            fullpulse_lowfreqbound = c
        if abs(freq1[c]-400) < 1:
            fullpulse_uppfreqbound = c
    # print(fullpulse_lowfreqbound)
    # print(fullpulse_uppfreqbound)


    # for c in range(len(freq2)):
    #     if abs(freq2[c] - 10) < 1:
    #         firsthalfpulse_lowfreqbound = c
    #     if abs(freq2[c] - 400) < 1:
    #         firsthalfpulse_uppfreqbound = c
    # # print(firsthalfpulse_lowfreqbound)
    # # print(firsthalfpulse_uppfreqbound)
    #
    # for c in range(len(freq3)):
    #     if abs(freq3[c] - 10) < 1:
    #         secondhalfpulse_lowfreqbound = c
    #     if abs(freq3[c] - 400) < 1:
    #         secondhalfpulse_uppfreqbound = c
    # print(secondhalfpulse_lowfreqbound)
    # print(secondhalfpulse_uppfreqbound)
    lowflag = 0
    for c in range(len(freq2)):
        if abs(freq2[c]) > 10 and lowflag == 0:
            firsthalfpulse_lowfreqbound = c
            lowflag = 1
        if abs(freq2[c]) > 400:
            firsthalfpulse_uppfreqbound = c
            break
    lowflag = 0
    for c in range(len(freq3)):
        if abs(freq3[c]) > 10 and lowflag == 0:
            secondhalfpulse_lowfreqbound = c
            lowflag = 1
        if abs(freq3[c]) > 400:
            secondhalfpulse_uppfreqbound = c
            break

    # print(firsthalfpulse_lowfreqbound)
    # print(firsthalfpulse_uppfreqbound)
    # print(secondhalfpulse_lowfreqbound)
    # print(secondhalfpulse_uppfreqbound)

    # print(len(freq2))
    # print(freq2[129])
    # firsthalfpulse_lowfreqbound = 4
    # secondhalfpulse_lowfreqbound = 4
    #
    # print(len(freq3))
    # print(hor[crossupperind[M]])
    # print(N)
    # print(freq3[9])
    # exit()

    spec_mom0 = 0
    spec_mom1 = 0
    spec_mom2 = 0
    spec_mom3 = 0
    spec_mom4 = 0
    spec_mom5 = 0

    lowspec_mom0 = 0
    lowspec_mom1 = 0
    lowspec_mom2 = 0
    lowspec_mom3 = 0
    lowspec_mom4 = 0
    lowspec_mom5 = 0

    upperspec_mom0 = 0
    upperspec_mom1 = 0
    upperspec_mom2 = 0
    upperspec_mom3 = 0
    upperspec_mom4 = 0
    upperspec_mom5 = 0
    #from 10-400 Hz

    # for i in range(17,664):
    for i in range(fullpulse_lowfreqbound, fullpulse_uppfreqbound):
        spec_mom0 = spec_mom0 + (math.pow(freq1[i],-1)*power_spec1[i])
        spec_mom1 = spec_mom1 + (math.pow(freq1[i], 1)*power_spec1[i])
        spec_mom2 = spec_mom2 + (math.pow(freq1[i], 2)*power_spec1[i])
        spec_mom3 = spec_mom3 + (math.pow(freq1[i], 3)*power_spec1[i])
        spec_mom4 = spec_mom4 + (math.pow(freq1[i], 4)*power_spec1[i])
        spec_mom5 = spec_mom5 + (math.pow(freq1[i], 5)*power_spec1[i])

    # for i in range(9, 332):
    for i in range(firsthalfpulse_lowfreqbound, firsthalfpulse_uppfreqbound):
        lowspec_mom0 = lowspec_mom0 + (math.pow(freq2[i], -1) * power_spec2[i])
        lowspec_mom1 = lowspec_mom1 + (math.pow(freq2[i], 1) * power_spec2[i])
        lowspec_mom2 = lowspec_mom2 + (math.pow(freq2[i], 2) * power_spec2[i])
        lowspec_mom3 = lowspec_mom3 + (math.pow(freq2[i], 3) * power_spec2[i])
        lowspec_mom4 = lowspec_mom4 + (math.pow(freq2[i], 4) * power_spec2[i])
        lowspec_mom5 = lowspec_mom5 + (math.pow(freq2[i], 5) * power_spec2[i])

    # for i in range(9, 332):
    for i in range(secondhalfpulse_lowfreqbound, secondhalfpulse_uppfreqbound):
        upperspec_mom0 = upperspec_mom0 + (math.pow(freq3[i], -1) * power_spec3[i])
        upperspec_mom1 = upperspec_mom1 + (math.pow(freq3[i], 1) * power_spec3[i])
        upperspec_mom2 = upperspec_mom2 + (math.pow(freq3[i], 2) * power_spec3[i])
        upperspec_mom3 = upperspec_mom3 + (math.pow(freq3[i], 3) * power_spec3[i])
        upperspec_mom4 = upperspec_mom4 + (math.pow(freq3[i], 4) * power_spec3[i])
        upperspec_mom5 = upperspec_mom5 + (math.pow(freq3[i], 5) * power_spec3[i])

    f2 = spec_mom0/spec_mom2
    f3 = spec_mom0/spec_mom3
    f4 = spec_mom0/spec_mom4
    f5 = spec_mom0/spec_mom5

    lowf2 = lowspec_mom0/lowspec_mom2
    lowf3 = lowspec_mom0/lowspec_mom3
    lowf4 = lowspec_mom0/lowspec_mom4
    lowf5 = lowspec_mom0/lowspec_mom5

    upperf2 = upperspec_mom0 / upperspec_mom2
    upperf3 = upperspec_mom0 / upperspec_mom3
    upperf4 = upperspec_mom0 / upperspec_mom4
    upperf5 = upperspec_mom0 / upperspec_mom5

    # print('Rep'+str(a+1))
    powsum = 0
    powarray = []

    lowpowsum = 0
    lowpowarray = []

    upperpowsum = 0
    upperpowarray = []

    for i in range(len(freq1)):
        powsum = powsum + power_spec1[i]
        powarray.append(powsum)

    for i in range(len(freq2)):
        lowpowsum = lowpowsum + power_spec2[i]
        lowpowarray.append(lowpowsum)

    for i in range(len(freq3)):
        upperpowsum = upperpowsum + power_spec3[i]
        upperpowarray.append(upperpowsum)

    mednum = powsum/2

    lowmednum = lowpowsum/2

    uppermednum = upperpowsum / 2

    # for i in powarray:
    #     if i > mednum:
    #         median = freq1[powarray.index(i)]
    #         print('median: ' + str(median))
    #         break
    #
    # for i in lowpowarray:
    #     if i > lowmednum:
    #         lowmedian = freq2[lowpowarray.index(i)]
    #         print('median: ' + str(lowmedian))
    #         break
    #
    # for i in upperpowarray:
    #     if i > uppermednum:
    #         uppermedian = freq3[upperpowarray.index(i)]
    #         print('median: ' + str(uppermedian))
    #         break


    meansumcombo = 0
    meansumpow = 0
    lowmeansumcombo = 0
    lowmeansumpow = 0
    uppermeansumcombo = 0
    uppermeansumpow = 0

    for i in range(len(freq1)):
        meansumcombo = meansumcombo + (freq1[i]*power_spec1[i])
        meansumpow = meansumpow + (power_spec1[i])
    for i in range(len(freq2)):
        lowmeansumcombo = lowmeansumcombo + (freq2[i] * power_spec2[i])
        lowmeansumpow = lowmeansumpow + (power_spec2[i])
    for i in range(len(freq3)):
        uppermeansumcombo = uppermeansumcombo + (freq3[i] * power_spec3[i])
        uppermeansumpow = uppermeansumpow + (power_spec3[i])

    mean = meansumcombo/meansumpow
    lowmean = lowmeansumcombo / lowmeansumpow
    uppermean = uppermeansumcombo / uppermeansumpow
    # print('mean: ' + str(mean))


    # print('Full')
    # if M ==74:
    #     for i in powarray:
    #         if i > mednum:
    #             median = freq1[powarray.index(i)]
    #             # print('median: ' + str(median))
    #             print(median)
    #             break
    #     print(hor[crossupperind[M]])
    #

    # for i in powarray:
    #     if i > mednum:
    #         median = freq1[powarray.index(i)]
    #         # print('median: ' + str(median))
    #         print(median)
    #         break
    # print(mean)
    # print(f2)
    # print(f3)
    # print(f4)
    # print(f5)
    #
    #
    # print('low half')
    # for i in lowpowarray:
    #     if i > lowmednum:
    #         lowmedian = freq2[lowpowarray.index(i)]
    #         # print('median: ' + str(lowmedian))
    #         print(lowmedian)
    #         break
    # #
    # print(lowmean)
    # print(lowf2)
    # print(lowf3)
    # print(lowf4)
    # print(lowf5)

    # print('upper half')
    # for i in upperpowarray:
    #     if i > uppermednum:
    #         uppermedian = freq3[upperpowarray.index(i)]
    #         # print('median: ' + str(uppermedian))
    #         print(uppermedian)
    #         break

    # print(uppermean)
    # print(upperf2)
    # print(upperf3)
    # print(upperf4)
    # print(upperf5)
    #
    # if N>60 and N<84:
    #     print(hor[crossupperind[M]])

    M=M+1
    N=N+1



    # if N == 8: ### for arm biodex set 1
    # if N == 21:  ### for arm biodex set 2
    #     # print(hor[crossupperind[M]])
    #     M = M+1
    #     N = N+1
    # if M == 61:  ### for arm biodex set 2
    #     # print(hor[crossupperind[M]])
    #     M = M+1
    #     N = N+1
    # if M == 69:  ### for arm biodex set 2
    #     # print(hor[crossupperind[M]])
    #     M = M+1
    #     N = N+1
    # if M == 74:  ### for arm biodex set 2
    #     # print(hor[crossupperind[M]])
    #     M = M+1
    #     N = N+1
    # if N == 83:  ### for arm biodex set 2
    #     # print(hor[crossupperind[M]])
    #     M = M+1
    #     N = N+1
    # M=M+2
    # N=N+2


exit()


plt.figure(2)
plt.plot(hor, emg_rec)

fig = plt.figure(3)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(sectionT, sectionpoints)

fig = plt.figure(4)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(x2, sectionpoints)

fig = plt.figure(5)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(freq1[4:], power_spec1[4:])

fig = plt.figure(6)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(lowhalfT, lowhalfsec)
fig = plt.figure(7)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(x3, lowhalfsec)
fig = plt.figure(8)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(freq2[4:], power_spec2[4:])


fig = plt.figure(9)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(upperhalfT, upperhalfsec)
fig = plt.figure(10)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(x4, upperhalfsec)
fig = plt.figure(11)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(freq3[4:], power_spec3[4:])

plt.show()