### CSV Extraction Code with help from: Jim Todd of Stack Overflow
from scipy import fftpack, signal
from matplotlib import pyplot as plt
import math
import numpy as np
import pandas as pd
import csv
import scipy.signal as sps
from scipy.signal import hilbert

from math import floor,log
# from .utils import _linear_regression

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

# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj01/Free/RF_Subj01_Free_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj01/Free/RF_Subj01_Free_Fatigue5_TR27.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj01/Hoka/RF_Subj01_Hoka_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj01/Hoka/RF_Subj01_Hoka_Fatigue5_TR34.csv', 'r') as csvfile:

# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj05/Hoka/RF_Subj05_Hoka_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj05/Hoka/RF_Subj05_Hoka_Fatigue5_TR28.csv', 'r') as csvfile:
#
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj08/Free/RF_Subj08_Free_Fatigue5_TR01.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj08/Free/RF_Subj08_Free_Fatigue5_TR24.csv', 'r') as csvfile:
filename = 'C:/Users/sword/Anaconda3/envs/exceltest/RF_Subj08/Free/RF_Subj08_Free_Fatigue5_TR24.csv'
with open(filename, 'r') as csvfile:

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

# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg_Fourth/Adam_Biodex_Leg4_TR04.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg_Fourth/Adam_Biodex_Leg4_TR05.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg_Fourth/Adam_Biodex_Leg4_TR06.csv', 'r') as csvfile:
# with open('C:/Users/sword/Anaconda3/envs/exceltest/NaEmAd/Adam_Biodex_Leg_Fourth/Adam_Biodex_Leg4_TR07.csv', 'r') as csvfile:


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
emg1 = df2.emg1
# emg3 = df2.emg3
# emg5 = df2.emg5
# emg7 = df2.emg7
# emg9 = df2.emg9
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
# emg1 = emg1[:964950] ##For adam biodex4 trial 6

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
# plt.show()

# emg23 = []
# for i in range(len(emg1)):
#     if emg1[i] < 0.005:
#         emg23.append(emg1[i])
#
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

# ## Auto Detect Musc On ##
# emg_rec = abs(emg1filt)
#
# slopes = []
#
#
# ynew = signal.savgol_filter(emg_rec,1501,2)
#
# cutoff_freq = 10  # ~20 Hz for movement according to emg book "Electromyography: Physiology, Engineering, and Noninvasive Apps"
# cut = cutoff_freq/nyq
# b, a = signal.butter(5, cut, btype='lowpass', analog=False)
# ynew2 = signal.filtfilt(b, a, ynew)
#
# ynew3 = signal.savgol_filter(ynew2,1501,2)
#


def katz_fd(x):
    """Katz Fractal Dimension.
    Parameters
    ----------
    x : list or np.array
        One dimensional time series
    Returns
    -------
    kfd : float
        Katz fractal dimension
    Notes
    -----
    The Katz Fractal dimension is defined by:
    .. math:: FD_{Katz} = \\frac{log_{10}(n)}{log_{10}(d/L)+log_{10}(n)}
    where :math:`L` is the total length of the time series and :math:`d`
    is the Euclidean distance between the first point in the
    series and the point that provides the furthest distance
    with respect to the first point.
    Original code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.
    References
    ----------
    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
           dimension algorithms. IEEE Transactions on Circuits and Systems I:
           Fundamental Theory and Applications, 48(2), 177-183.
    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
           the computation of EEG biomarkers for dementia." 2nd International
           Conference on Computational Intelligence in Medicine and Healthcare
           (CIMED2005). 2005.
    Examples
    --------
    Katz fractal dimension.
        # >>> import numpy as np
        # >>> from entropy import katz_fd
        # >>> np.random.seed(123)
        # >>> x = np.random.rand(100)
        # >>> print(katz_fd(x))
            5.1214
    """

    x = np.array(x)
    dists = np.abs(np.ediff1d(x))
    ll = dists.sum()
    ln = np.log10(np.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = np.max(np.abs(aux_d[1:]))
    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))


def _linear_regression(x, y):
    """Fast linear regression using Numba.
    Parameters
    ----------
    x, y : ndarray, shape (n_times,)
        Variables
    Returns
    -------
    slope : float
        Slope of 1D least-square regression.
    intercept : float
        Intercept
    """
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept







def _higuchi_fd(x, kmax):
    """Utility function for `higuchi_fd`.
    """
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi


# for i in range(len(emg1filt)//750):
#     emg_sec = emg1filt[0+(750*i):7500+(750*i)]
#     frac1 = _higuchi_fd(emg_sec,6)
#     print(frac1)

# print(katz_fd(emg1filt))
for i in range(len(emg1filt)//750):
    emg_sec = emg1filt[0+(750*i):7500+(750*i)]
    frac2 = katz_fd(emg_sec)
    print(frac2)