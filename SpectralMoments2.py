### CSV Extraction Code with help from: Jim Todd of Stack Overflow
from scipy import fftpack, signal
from matplotlib import pyplot as plt
import math
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

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Right_Solo/Dimitrov_Session01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Right_Solo/Dimitrov_Session02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Right_Solo/Dimitrov_Session03.csv', 'r') as csvfile:

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
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg/Adam_Biodex_Leg02.csv','r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg2/Adam_Biodex_Leg_Second01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg2/Adam_Biodex_Leg_Second02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg2/Adam_Biodex_Leg_Second03.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg3/Biodex_Leg_Test01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg3/Biodex_Leg_Test02.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg3/Biodex_Leg_Test03.csv', 'r') as csvfile:



# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg_Fourth/Adam_Biodex_Leg4_TR04.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg_Fourth/Adam_Biodex_Leg4_TR05.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg_Fourth/Adam_Biodex_Leg4_TR06.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg_Fourth/Adam_Biodex_Leg4_TR07.csv', 'r') as csvfile:
with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj08/Free/RF_Subj08_Free_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj08/Free/RF_Subj08_Free_Fatigue5_TR24.csv', 'r') as csvfile:

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
emg1 = df2.emg2
print(len(hor))

# emgtemp = emg1[96000:837000] ##For adam running trial (biodex second 3)
# emg1 = []

# emgtemp = emg1[84900:924000] ##For adam biodex3 trial 1
# emg1 = []

# emgtemp = emg1[79500:1212000] ##For adam biodex3 trial 2
# emg1 = []

# emgtemp = emg1[94125:] ##For adam biodex4 trial 4
# emg1 = []

# emgtemp = emg1[:964950] ##For adam biodex4 trial 6
# emg1 = []

# emg1 = emg1[292500:658500]  ##For adam biodex right leg 1
# hor = hor[292500:658500]

# emg1 = emg1[96000:837000]  ##For adam running trial (biodex second 3)
# hor = hor[96000:837000]
#
# for i in emgtemp:
#     emg1.append(i)
# hor = np.arange(0, (len(emg1)-0.5)/samplerate, 1/samplerate)

# plt.figure(1)
# plt.plot(hor, emg1)
# plt.title('Biceps')

# emg23 = []
# for i in range(len(emg1)):
#     if emg1[i] < 0.005:
#         emg23.append(emg1[i])

# print('we did it bois')
# hor2 = np.arange(0, (len(emg23)-0.5)/samplerate, 1/samplerate)  #getting the time domain in seconds
# emg1 = emg23
# hor = hor2
# print(len(hor))


cutoff_freq = 20  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='highpass', analog=False)
emg1high = signal.filtfilt(b, a, emg1)

cutoff_freq = 400  # ~500 Hz according to the emg book
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='lowpass', analog=False)
emg1filt = signal.filtfilt(b, a, emg1high)

## Auto Detect Musc On ##
emg_rec = abs(emg1filt)

slopes = []


ynew = signal.savgol_filter(emg_rec,1501,2)

cutoff_freq = 10  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
cut = cutoff_freq/nyq
b, a = signal.butter(5, cut, btype='lowpass', analog=False)
ynew2 = signal.filtfilt(b, a, ynew)

ynew3 = signal.savgol_filter(ynew2,1501,2)


peaks, props = signal.find_peaks(ynew3, width=(500,2000), rel_height=0.8)
print("len of peaks: " + str(len(peaks)))

pulses_beginT = []
pulses_endT = []
pulses_begin = []
pulses_end = []
pulses_begin_ind = []
pulses_end_ind = []
for i in range(len(peaks)):
    pulse_sample_start = peaks[i] - (math.floor(props['widths'][i]/2))
    pulse_sample_end = peaks[i] + (math.floor(props['widths'][i]/2))
    pulses_begin_ind.append(pulse_sample_start)
    pulses_end_ind.append(pulse_sample_end)
    pulses_beginT.append(pulse_sample_start/1500)
    pulses_endT.append(pulse_sample_end/1500)
    pulses_begin.append(ynew3[pulse_sample_start])
    pulses_end.append(ynew3[pulse_sample_end])

ymean = np.average(ynew)
yml = []
for i in hor:
    yml.append(ymean*0.9)

yml25 = []
for i in hor:
    yml25.append(ymean*0.5)



M = 0
N = 1
lowcut = 10
highcut = 400
for a in range((len(peaks))):

    sectionpoints = []
    sectionT = []
    for i in range(M, N):
        sectionpoints.extend(emg_rec[pulses_begin_ind[i]:pulses_end_ind[i]])
        # sectionT.extend(hor[crossupperind[i]:crosslowerind[i]])
        sectionT.extend(hor[pulses_begin_ind[i]:pulses_end_ind[i]])

    x2 = np.linspace(0, len(sectionpoints), len(sectionpoints))
    x2 = x2 / 1500

    sectionpoints_array = np.asarray(sectionpoints)

    freq1, power_spec1 = signal.periodogram(sectionpoints_array, samplerate)

    summ1 = 0
    summ2 = 0

    lowflag = 0
    for c in range(len(freq1)):
        if abs(freq1[c]) > 10 and lowflag == 0:
            fullpulse_lowfreqbound = c
            lowflag = 1
        if abs(freq1[c]) > 400:
            fullpulse_uppfreqbound = c
            break

    spec_mom0 = 0
    spec_mom1 = 0
    spec_mom2 = 0
    spec_mom3 = 0
    spec_mom4 = 0
    spec_mom5 = 0

    for i in range(fullpulse_lowfreqbound, fullpulse_uppfreqbound):
        spec_mom0 = spec_mom0 + (math.pow(freq1[i], -1) * power_spec1[i])
        spec_mom1 = spec_mom1 + (math.pow(freq1[i], 1) * power_spec1[i])
        spec_mom2 = spec_mom2 + (math.pow(freq1[i], 2) * power_spec1[i])
        spec_mom3 = spec_mom3 + (math.pow(freq1[i], 3) * power_spec1[i])
        spec_mom4 = spec_mom4 + (math.pow(freq1[i], 4) * power_spec1[i])
        spec_mom5 = spec_mom5 + (math.pow(freq1[i], 5) * power_spec1[i])

    f2 = spec_mom0 / spec_mom2
    f3 = spec_mom0 / spec_mom3
    f4 = spec_mom0 / spec_mom4
    f5 = spec_mom0 / spec_mom5

    powsum = 0
    powarray = []

    for i in range(len(freq1)):
        powsum = powsum + power_spec1[i]
        powarray.append(powsum)

    mednum = powsum / 2

    meansumcombo = 0
    meansumpow = 0

    for i in range(len(freq1)):
        meansumcombo = meansumcombo + (freq1[i]*power_spec1[i])
        meansumpow = meansumpow + (power_spec1[i])

    mean = meansumcombo / meansumpow

    # for i in powarray:
    #     if i > mednum:
    #         median = freq1[powarray.index(i)]
    #         # print('median: ' + str(median))
    #         print(median)
    #         break
    print(mean)
    # print(f2)
    # print(f5)

    M = M + 1
    N = N + 1


fig = plt.figure(1)
ax0 = fig.add_subplot(1,1,1)
ax0.plot(hor, emg_rec)

ax0.plot(hor, yml, color = 'khaki')
ax0.plot(hor, yml25, color = 'purple')

ax0.plot(hor, ynew3, color = 'purple')


ax0.plot(pulses_beginT, pulses_begin, marker = "x",color = 'yellow', linestyle = "None")
ax0.plot(pulses_endT, pulses_end, marker = "x",color = 'yellow', linestyle = "None")
#
# peakT = []
# for i in peaks:
#     peakT.append(hor[i])
# ax0.plot(peakT, ynew_peaks, marker = "x",color = 'yellow', linestyle = "None")
# ax0.plot(pulse_begin, ybeg, marker = "x",color = 'yellow', linestyle = "None")
# ax0.plot(pulse_end, yend, marker = "x",color = 'yellow', linestyle = "None")





plt.show()
