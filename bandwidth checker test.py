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
with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR09.csv', 'r') as csvfile:     #Adam Arm Dyn Set 1
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
df.columns = ['frames', 'subframes', 'blank', 'NU', 'NU', 'NatTA', 'NU', 'NatBic', 'NatTri', 'NatGastLat', 'NatGastMed', 'EmilyBic', 'EmilyTri', 'EmilyGastMed', 'EmilyGastLat', 'EmilyTA', 'AdamBic', 'AdamTri', 'unused6', 'unused7', 'unused8', 'blank2']
# df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused4', 'unused5', 'unused6', 'unused7', 'unused8', 'blank2']
# df.columns = ['frames', 'subframes', 'blank', 'RGastLat', 'LGastLat', 'RTibAnt', 'LTibAnt', 'RBicFem', 'LBicFem', 'RRecFem', 'LRecFem', 'RGastMed', 'LGastMed', 'unused1', 'unused2', 'unused3', 'unused7', 'unused8', 'blank2']
df2 = df.drop(['frames', 'subframes', 'blank', 'unused7', 'unused8', 'blank2'], axis = 1)
#print(df2)
df2 = df2.astype(np.float)
print(len(df))
hor = np.arange(0, (len(df)-0.5)/samplerate, 1/samplerate)  #getting the time domain in seconds
emg1 = df2.AdamBic
print(len(hor))
plt.figure(1)
plt.plot(hor, emg1)
plt.title('Biceps')

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

    elif (ynew3[i] < (yml25[0]) and ynew3[i - 1] > (yml25[0])):
        if pointswitch == 1:
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


print('pulses: ' + str(len(crosslowerind)))
M = 0
N = 1
lowcut = 10
highcut = 400
for a in range(len(crosslowerind)):

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

    unweightedpowsum = 0
    for i in range(17, 664):
        unweightedpowsum = unweightedpowsum + power_spec1[i]


    bwcheck = 0.95 * unweightedpowsum
    loopbw = 0
    n = 17
    while loopbw < bwcheck:
        loopbw = loopbw + power_spec1[n]
        n = n + 1
    # print('95% power BW')
    # print(freq1[n])
    # print('freq index: ' + str(n))
    # print('')

    bwcheck = 0.9 * unweightedpowsum
    loopbw = 0
    n = 17
    while loopbw < bwcheck:
        loopbw = loopbw + power_spec1[n]
        n = n + 1
    # print('90% power BW')
    # print(freq1[n])
    # print('freq index: ' + str(n))
    # print('')

    bwcheck = 0.8 * unweightedpowsum
    loopbw = 0
    n = 17
    while loopbw < bwcheck:
        loopbw = loopbw + power_spec1[n]
        n = n + 1
    # print('80% power BW')
    # print(freq1[n])
    # print('freq index: ' + str(n))
    # print('')

    bwcheck = 0.3 * unweightedpowsum
    loopbw = 0
    n = 17
    while loopbw < bwcheck:
        loopbw = loopbw + power_spec1[n]
        n = n + 1
    # print('80% power BW')
    print(freq1[n])
    # print('freq index: ' + str(n))
    # print('')


    M=M+1
    N=N+1



exit()











plt.figure(2)
plt.plot(hor, emg_rec)

fig = plt.figure(3)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(sectionT, sectionpoints)

fig = plt.figure(4)
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(x2, sectionpoints)


plt.show()