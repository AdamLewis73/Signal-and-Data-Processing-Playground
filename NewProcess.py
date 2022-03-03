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
with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/ArmandCalf_TR17.csv', 'r') as csvfile:     #Emily Calf Raise Dyn Set 1 ########
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
emg1 = df2.EmilyTA
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

# print(slop[3200:3400])
###########
###########

###########
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
    elif (ynew3[i] < yml25[0] and ynew3[i - 1] > yml25[0]):
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
                tempwidth = crossTlower[0] - crossTupper[0]
                # print(tempwidth)
                widtharray.append(tempwidth)
                firstwidthflag = 1
            else:
                tempwidth = crossTlower[-1] - crossTupper[-1]
                # print(tempwidth)
                if (tempwidth > (0.5* np.median(widtharray)) and tempwidth < (1.75 * np.median(widtharray))):
                    widtharray.append(tempwidth)
                else:
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


# trylist = [1,2,3,4,5]
# print(trylist[-1])
# trylist.append(8)
# print(trylist[-1])

M = 40
N = 45
# N = len(crosslowerind)
sectionpoints = []
sectionT = []

for i in range(M,N):
    sectionpoints.extend(emg_rec[crossupperind[i]:crosslowerind[i]])
    sectionT.extend(hor[crossupperind[i]:crosslowerind[i]])

x2 = np.linspace(0, len(sectionpoints), len(sectionpoints))
x2 = x2/1500

# checker3=0
# crossp = []
# crossT = []
# crossind = []
# last = emg_rec[0]
# firsttime = 0
# widtharray = []
# firsttime2 = 0
# for i in range(0,len(slop)):
#     # if ynew3[i]< yml25[0]:
#     if emg_rec[i] < yml[0]:
#         if last > yml[0]:
#             checker3+=1
#             if firsttime == 0:
#                 tempcrosstime = hor[i]
#                 tempcrossind = i
#                 firsttime = 1
#             elif (hor[i]-tempcrosstime) < 0.040553333333333145:
#                 tempwidth = i-tempcrossind
#                 if firsttime2 == 0:
#                     firsttime2 = 1
#                     widtharray.append(tempwidth)
#                 elif tempwidth > (0.75*np.median(widtharray)):
#                     # crossp.extend(yml[tempcrossind:(i-1)])
#                     crossp.extend(emg_rec[tempcrossind:(i - 1)])
#                     crossT.extend(hor[tempcrossind:(i-1)])
#                     widtharray.append(tempwidth)
#                 # crossind.append((tempcrosstime:i-1))
#
#             tempcrosstime = hor[i]
#             tempcrossind = i
#     elif emg_rec[i] > yml[0]:
#         if last < yml[0]:
#             checker3+=1
#             if firsttime == 0:
#                 tempcrosstime = hor[i]
#                 tempcrossind = i
#                 firsttime = 1
#             elif (hor[i] - tempcrosstime) < 0.040553333333333145:
#                 tempwidth = i - tempcrossind
#                 if firsttime2 == 0:
#                     firsttime2 = 1
#                     widtharray.append(tempwidth)
#                 elif tempwidth > (0.75 * np.median(widtharray)):
#                     # crossp.extend(yml[tempcrossind:(i-1)])
#                     crossp.extend(emg_rec[tempcrossind:(i - 1)])
#                     crossT.extend(hor[tempcrossind:(i-1)])
#                 # crossind.append((tempcrosstime:i-1))
#             tempcrosstime = hor[i]
#             tempcrossind = i

    # last = emg_rec[i]





# print(crossp)
# print(crossT)
                     # This is the backup
# for i in range(0,len(slop)):
#     # if ynew3[i]< yml25[0]:
#     if emg_rec[i] < yml[0]:
#         if last > yml[0]:
#             checker3+=1
#             crossp.append(yml[i-1])
#             crossT.append(hor[i-1])
#             crossind.append((i-1))
#     elif emg_rec[i] > yml[0]:
#         if last < yml[0]:
#             checker3+=1
#             crossp.append(yml[i-1])
#             crossT.append(hor[i-1])
#             crossind.append((i-1))
#     last = emg_rec[i]





# for i in range(0,len(slop)):
#     # if ynew3[i]< yml25[0]:
#     if
#     if emg_rec[i] < yml[0]:
#         if last > yml[0]:
#             checker3+=1
#             crossp.append(yml[i-1])
#             crossT.append(hor[i-1])
#             crossind.append((i-1))
#     elif emg_rec[i] > yml[0]:
#         if last < yml[0]:
#             checker3+=1
#             crossp.append(yml[i-1])
#             crossT.append(hor[i-1])
#             crossind.append((i-1))
#     last = emg_rec[i]


# print(checker3)
# difft = []
# for i in range(0,600):
#     crossdiff = crossT[i+1]-crossT[i]
#     difft.append(crossdiff)
# crossTavg = np.average(difft)
# crossMax = np.max(difft)
# print(crossTavg)
# print(np.max(difft))
# print(np.min(difft))
# print(crossTavg + 1.5 * np.max(difft))
# print(len(hor))


# crosskeep = []
# crosskeepT = []
# for i in range(1, len(crossT)):
#     if (crossT[i]-crossT[i-1])< (crossTavg + 1.5 * np.max(difft)):
#         # crosskeep.extend(emg_rec[(i-1):i])
#         # crosskeepT.extend(crossT[(i-1):i])
#         crosskeep.append(emg_rec[i-1])
#         crosskeepT.append(crossT[i-1])


#########123456############
# firstcross = 0
# keepme = []
# keepmeT = []
# crosspoint = 0
# crosspointT = []
# firstpoint = 0
# ck1 = 0
# ck2 = 0
# firstwidth =0
# timewidth = []
# crosswidth = 0
# temparray = []
# temparrayT = []
# tem1 = []
# temt1 = []
# tem1in = []
# temt1in = []
# temt1in.append(0)
# temporary1 = 0
# temporary2 = 0
# tem1.append(emg_rec[0])
# temt1.append(hor[0])
# for i in range(1, len(hor)):               #### Add a flag for crossing once and then crossing again, possibly changing this whole loop section
#     if ((emg_rec[i] < yml[0] and emg_rec[i-1]> yml[0]) or (emg_rec[i] > yml[0] and emg_rec[i-1] < yml[0])):
#         if firstcross == 0:
#             crosspoint = i-1
#             # crosspointT.append[i-1]
#             firstpoint = i-1
#             firstcross = 1
#             ck1 += 1
#         elif firstcross == 1:
#             ck2 += 1
#             crosswidth = hor[i-1]-hor[crosspoint]
#             # if i<20000:
#             if crosswidth > (crossTavg + 1.5 * np.max(difft)):
#                 tem1.append(emg_rec[crosspoint])
#                 temt1.append(hor[crosspoint])
#                 # temporary1 = np.where(emg_rec[crosspoint])
#                 temt1in.append(crosspoint)
#
#                 # if crosswidth>0.025:
#
#                 # if crosswidth > (crossTavg + 1.5 * np.max(difft)):
#                 #     tem1.append(emg_rec[crosspoint])
#                 #     temt1.append(hor[crosspoint])
#                 #     # temporary1 = np.where(emg_rec[crosspoint])
#                 #     temt1in.append(crosspoint)
#
#
#                 # print(crosswidth)
#             if crosswidth < (crossTavg + 1.5 * np.max(difft)):
#                 keepme.extend(emg_rec[crosspoint:(i-1)])
#                 keepmeT.extend(hor[crosspoint:(i-1)])
#             elif crosswidth > (crossTavg + 1.5 * np.max(difft)):
#                 # firstcross =
#                 # crosspoint = i - 1
#                 # crosspointT.append[i-1]
#                 # width = hor[i-1]-hor[firstpoint]
#                 width = hor[crosspoint]-hor[firstpoint]
#                 if firstwidth == 0:
#                     timewidth.append(width)
#                     firstwidth = 1
#                 elif firstwidth == 1:
#                     if ((width > (0.75 * np.median(timewidth))) and (width <(1.5*np.median(timewidth)))):
#                         timewidth.append(width)
#                 firstpoint = i - 1
#             crosspoint = i-1
#             # crosspointT.append[i - 1]
#
# #
# # print(ck1)
# # print(ck2)
# # # print(tem1in)
# # print(temt1in)
# print("split")
# tempwidth = temt1[1] - temt1[0]
# # print(tempwidth)
# widtharray = []
# widtharray.append(tempwidth)
# keepme1 = []
# keepme1T = []
# ck3 = 0
# for i in range(1, len(tem1)):
#     tempwidth = temt1[i] - temt1[i-1]
#     # if i <= 3:
#         # print(0.75 * np.median(widtharray))
#         # print(tempwidth)
#         # print(temt1[i])
#     # if ((tempwidth > 0.75* np.median(widtharray)) and (np.max(emg_rec[(i-1):i])>yml[0])):     ### does not work because of emg rec indexing
#     if tempwidth > 0.75 * np.median(widtharray):
#         ck3 +=1
#         # print("i equals" + str(i))
#         widtharray.append(tempwidth)
#         # keepme1.extend(emg_rec[temt1[i-1]:temt1[i]])
#         # keepme1T.extend(hor[temt1[i - 1]:temt1[i]])
#         # if i < len(temt1in):
#             # print(temt1in[i-1])
#             # print('we did it')
#         # if i == 1:
#         #     keepme1.extend(emg_rec[0:temt1in[i]])
#         #     keepme1T.extend(hor[0:temt1in[i]])
#         # else:
#         # print(emg_rec[temt1in[i-1]:temt1in[i]])
#         # print(hor[temt1in[i - 1]:temt1in[i]])
#         keepme1.extend(emg_rec[temt1in[i-1]:temt1in[i]])
#         keepme1T.extend(hor[temt1in[i - 1]:temt1in[i]])
#
#######123456##########

# print(ck3)
# print(widtharray)
# print(keepme1)
# print(keepme1T)
        ###set flag variable to 1
    # else:
    #     if ####the previous section was good, flag variable is 1
    #
    #         temt1.remove(i) ### maybe just append or not append emg data and times
    #         ###set flag variable to 0
    #     elif ###previous section not good

# ##############################################################
# firstcross = 0
# keepme = []
# keepmeT = []
# crosspoint = 0
# firstpoint = 0
# ck1 = 0
# firstwidth =0
# timewidth = []
# crosswidth = 0
# temparray = []
# temparrayT = []
# for i in range(1, len(hor)):               #### Add a flag for crossing once and then crossing again, possibly changing this whole loop section
#     if ((emg_rec[i] < yml[0] and emg_rec[i-1]> yml[0]) or (emg_rec[i] > yml[0] and emg_rec[i-1] < yml[0])):
#         if firstcross == 0:
#             firstpoint = i-1      ###### SET A FLAG FOR EVERY CROSS.....ACTUALLY MAYBE JUST
#             crosspoint = i-1   ### Need a variable for first point and each cross point
#             firstcross = 1
#         elif firstcross == 1:        ####DO THIS
#             # temparray.extend(emg_rec[crosspoint:i])
#             # temparrayT.extend(hor[crosspoint:i])
#
#
#             crosswidth = hor[i]-hor[crosspoint]
#             crosspoint = i
#             width = temparrayT[-1]-temparrayT[0]
#
#             if crosswidth < (crossTavg + 1.5*np.max(difft)):
#                 # keepme.append(emg_rec[crosspoint:i])
#                 # keepmeT.append(hor[crosspoint:i])
#                 ck1 +=1
#             # print(ck1)
#             elif firstwidth ==0:
#                 timewidth.append(width)
#                 keepme.extend(temparray)
#                 keepmeT.extend(temparrayT)
#                 firstwidth = 1
#             elif ((width > (0.75 * np.median(timewidth))) and (width <(1.5*np.median(timewidth)))):
#                 firstcross = 0
#                 timewidth.append(width)
#                 keepme.extend(temparray)
#                 keepmeT.extend(temparrayT)
#
# print(ck1)
# print(keepme)


# for i in range(1, len(hor)):               #### Add a flag for crossing once and then crossing again, possibly changing this whole loop section
#     if ((emg_rec[i] < yml[0] and emg_rec[i-1]> yml[0]) or (emg_rec[i] > yml[0] and emg_rec[i-1] < yml[0])):
#         if flagcross == 1:
#             timewidth = hor[i]-hor[crosspoint]
#             if timewidth < (crossTavg + 1.5*np.max(difft)):
#                 # keepme.append(emg_rec[crosspoint:i])
#                 # keepmeT.append(hor[crosspoint:i])
#                 keepme.extend(emg_rec[crosspoint:i])
#                 keepmeT.extend(hor[crosspoint:i])
#                 ck1 +=1
#                 # print(ck1)
#             else:
#
#             flagcross = 0
#
#         elif flagcross == 0:
#             crosspoint = i
#             flagcross = 1
#
# print(ck1)
# print(keepme)







# last = emg_rec[0]
# keeper = []
# keeperT = []
# flag = 0
# firsttime = 0
# pulsewidths = []
# pulsepower = []
# flag_powergap = 0
# flagfirst = 0
# ck1 = 0
# ck2 = 0
# for i in range(1, len(hor)):                                    ####CHECK IF IT CROSS THE YML LINE YOU IDIOT
#     diffcheck = hor[i]-hor[i-1]
#     if emg_rec[i] < yml[0]:
#         if last > yml[0]:
#             if diffcheck < (crossTavg + 2*np.max(difft)):
#
#                 ck1 += 1
#                 if flagfirst == 0:
#                     flag = i
#                     flagfirst = 1
#                     keeper.append(emg_rec[i-1])
#                     keeperT.append(hor[i-1])
#
#                 keeper.append(emg_rec[i])
#                 keeperT.append(hor[i])
#                 flag_powergap = 1
#
#
#         last = emg_rec[i]
#
#     elif emg_rec[i] > yml[0]:
#         if last < yml[0]:
#             if diffcheck < (crossTavg + 2 * np.max(difft)):
#                 ck1 += 1
#                 if flagfirst == 0:
#                     flag = i
#                     flagfirst = 1
#                     keeper.append(emg_rec[i - 1])
#                     keeperT.append(hor[i - 1])
#
#                 keeper.append(emg_rec[i])
#                 keeperT.append(hor[i])
#                 flag_powergap = 1
#         last = emg_rec[i]
#     elif flag_powergap == 1:
#         ck2 += 1
#         width = hor[i]-hor[flag]
#         # timePower = hor[flag:i]
#         # widthPower = emg_rec[flag:i]
#         flagfirst = 0
#         if firsttime == 0:                                  ## only happens once the whole program
#             pulsewidths.append(width)
#             print(width)
#             # pulsepower.append(widthPower
#             firsttime = 1
#         elif (width >(0.75*np.median(pulsewidths)) and width < (1.5*np.median(pulsewidths))):
#             pulsewidths.append(width)
#             # pulsepower.append(widthPower)
# print(pulsewidths)
# print(ck1)
# print(ck2)
# print(max(ynew))

#121 contractions with 6 possible double humps at bottom (so possibly will read around 127)

# if ynew3



# rising = []
# risingT = []
# compar = 0
# checker = 0
# for i in hor:
#     if ynew[i] > 0:
#         if (ynew[i] < ynew[i-1] and ynew[i-1] > ynew[i-2]):
#             rising.append(ynew[i-1])
#             risingT.append(hor[i-1])
#     # compar = ynew[i]
#     # checker = hor[i]


# peaks = []
# peaksT = []
# for i in hor:
#     if slopes[i]>0:
#         peaks.append(slopes[i])
#         peaksT.append(hor[i])
#     elif
# print(len(slopes))
# N = 10
# cumsum, moving_aves = [0], []
#
# for i, x in enumerate(ynew, 1):
#     cumsum.append(cumsum[i-1] + x)
#     if i>=N:
#         moving_ave = (cumsum[i] - cumsum[i-N])/N
#         #can do stuff with moving_ave here
#         moving_aves.append(moving_ave)




# low_pass = 20
# low_pass = low_pass / 1500
# b2, a2 = signal.butter(4, low_pass, btype='lowpass')
# emg_envelope = signal.filtfilt(b2, a2, emg_rec)
#
# emg_envelope =
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
for i in range(0,20):
    print(isopowavg[i])

# exit()
###################

plt.figure(6)
plt.specgram(sectionpoints_array,Fs=samplerate,NFFT=128,noverlap=0)

freq1, power_spec1 = signal.periodogram(sectionpoints_array, samplerate)

meantest1 = sum(freq1*power_spec1)/sum(power_spec1)
print('Mean frequency is: ' + str(round(meantest1, 2))+' Hz')




plt.show()





















#
# ####  INSERT ARRAY SPLICING HERE  ####
# # #
# # emg1temp = emg1filt[:33495]
# # emg2temp = emg1filt[34305:38250]
# # emg3temp = emg1filt[39495:86700]
# # emg4temp = emg1filt[88200:307245]
# # emg5temp = emg1filt[308970:]
# # # emg6temp = emg1filt[362400:]
# # # # emg7temp = emg1filt[:]
# # # # emg8temp = emg1filt[:]
# # # #
# # emg1filt = np.concatenate([emg1temp, emg2temp, emg3temp, emg4temp, emg5temp])
# # hor = np.arange(0, (len(emg1filt)-0.5)/samplerate, 1/samplerate)
#
# ####    END SPLICING CODE   ####
# # emg1filt = emg1filt[:12500]
# # hor = hor[:12500]
# # emg1filt = emg1filt[12501:25000]
# # hor = hor[12501:25000]
# # emg1filt = emg1filt[25001:37500]
# # hor = hor[25001:37500]
# # emg1filt = emg1filt[37501:50000]
# # hor = hor[37501:50000]
# # emg1filt = emg1filt[50001:62500]
# # hor = hor[50001:62500]
# # emg1filt = emg1filt[62501:75000]
# # hor = hor[62501:75000]
# # emg1filt = emg1filt[75001:87500]
# # hor = hor[75001:87500]
# # emg1filt = emg1filt[87501:100000]
# # hor = hor[87501:100000]
# # emg1filt = emg1filt[100001:112500]
# # hor = hor[100001:112500]
# # emg1filt = emg1filt[112501:]
# # hor = hor[112501:]
# # #
# # print(len(emg1filt))
# # n=7
# # m=n+1
# #
# # emg1filt = emg1filt[30001*n:m*30000]
# # hor = hor[30001*n:m*30000]
# #
# # print(len(emg1filt))
# # n=0
# # m=n+1
# #
# # emg1filt = emg1filt[15001*n:m*15000]
# # hor = hor[15001*n:m*15000]
#
#
# plt.figure(2)
# plt.plot(hor, emg1filt)
# plt.title('Biceps Filtered')
#
# freq1, power_spec1 = signal.periodogram(emg1filt, samplerate)
#
# meantest1 = sum(freq1*power_spec1)/sum(power_spec1)
# print('Biceps mean frequency is: ' + str(round(meantest1, 2))+' Hz')
#
# plt.figure(3)
# plt.semilogy(freq1, power_spec1)
# plt.ylim([1e-17, 1e-7])
# plt.title('Biceps: ' + str(round(meantest1, 2))+' Hz')
#
# plt.figure(4)
# f1,t1, Sxx1 = sps.spectrogram(emg1filt,samplerate,('hamming'),128,0, scaling= 'density')
# plt.pcolormesh(t1,f1,Sxx1)
# plt.ylim((0,300))
# plt.title('Biceps')
#
# ###ISO POWER###
# # print(Sxx1.shape)
# # print(len(Sxx1))
# # isopowavg = np.average(Sxx1[:20,:], axis = 1)
# # # print(isopowavg.shape)
# # for i in range(0,20):
# #     print(isopowavg[i])
#
# # exit()
# ###################
#
# plt.figure(5)
# plt.specgram(emg1filt,Fs=samplerate,NFFT=128,noverlap=0)
#
# maxes1 = np.amax(Sxx1, axis=0)
# maxesind1 = np.argmax(Sxx1, axis=0)
# fval1 = [f1[i] for i in maxesind1]
#
# plt.show()
#
# fig = plt.figure(6)
# ax12 = fig.add_subplot(1,1,1, label = '1')
# ax12.xaxis.tick_top()
# ax12.yaxis.tick_right()
# # print(min(maxes1))
# plt.plot(t1,maxes1, color = 'red')
# ax1 = fig.add_subplot(1,1,1, label = '2', frame_on = False)
# plt.plot(t1,fval1)
# plt.title('Biceps')
#
# maxfiltind1 = [i for i,ele in enumerate(maxes1) if ele>(min(maxes1)*15)]
# maxfilt1 = [ele for i,ele in enumerate(maxes1) if ele>(min(maxes1)*15)]  ##Test - change val as neede
#
# freqchan = []
# freqchan = [Sxx1[:,i] for i in maxfiltind1]
# freqchan = np.reshape(freqchan, (len(f1), len(maxfiltind1)))
# freqpow = freqchan[:20,:]
# freqpow2 = freqpow.mean(axis=1)
#
# # print(freqchan)
# # print(freqpow)
# # print(freqpow2)
# # print(freqpow2.shape)
#
# print("         <  7.8125 Hz: ", freqpow2[0])
# print("7.8125  -   15.625 Hz: ", freqpow2[1])
# print("15.625 -   23.4375 Hz: ", freqpow2[2])
# print("23.4375  -   31.25 Hz: ", freqpow2[3])
# print("31.25  -   39.0625 Hz: ", freqpow2[4])
# print("39.0625  -  46.875 Hz: ", freqpow2[5])
# print("46.875  -  54.6875 Hz: ", freqpow2[6])
# print("54.6875    -  62.5 Hz: ", freqpow2[7])
# print("62.5    -  70.3125 Hz: ", freqpow2[8])
# print("70.3125  -  78.125 Hz: ", freqpow2[9])
# print("78.125  -  85.9375 Hz: ", freqpow2[10])
# print("85.9375   -  93.75 Hz: ", freqpow2[11])
# print("93.75  -  101.5625 Hz: ", freqpow2[12])
# print("101.5625 - 109.375 Hz: ", freqpow2[13])
# print("109.375 - 117.1875 Hz: ", freqpow2[14])
# print("117.1875   -   125 Hz: ", freqpow2[15])
# print("125   -   132.8125 Hz: ", freqpow2[16])
# print("132.8125 - 140.625 Hz: ", freqpow2[17])
# print("140.625 - 148.4375 Hz: ", freqpow2[18])
# print("148.4375 -  156.25 Hz: ", freqpow2[19])
#
# print(freqpow2[0])
# print(freqpow2[1])
# print(freqpow2[2])
# print(freqpow2[3])
# print(freqpow2[4])
# print(freqpow2[5])
# print(freqpow2[6])
# print(freqpow2[7])
# print(freqpow2[8])
# print(freqpow2[9])
# print(freqpow2[10])
# print(freqpow2[11])
# print(freqpow2[12])
# print(freqpow2[13])
# print(freqpow2[14])
# print(freqpow2[15])
# print(freqpow2[16])
# print(freqpow2[17])
# print(freqpow2[18])
# print(freqpow2[19])
# # plt.show()
#
# saveind = []
# saveind = np.empty(shape=(0,2))
# for index,k in np.ndenumerate(Sxx1):
#     for val in maxfilt1:
#         if k == val:
#             saveind = np.append(saveind, index)
# saveind = np.reshape(saveind,(len(saveind)//2,2))
# saveind = saveind[saveind[:,1].argsort()]
# saveind = np.delete(saveind,1,1)
# saveind = saveind.astype(int)
#
# fval1 = [f1[i] for i in saveind]
# fval1 = np.concatenate(fval1,axis = 0)
# avg1 = np.average(fval1)
# print('RGastLat Avg: ', avg1)
# # csvfile.close()
# # plt.show()
