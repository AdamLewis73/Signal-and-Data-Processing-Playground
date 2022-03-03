from scipy import fftpack,signal
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

f1 = 10
samples = 5000
w = 2.*np.pi*f1
time = np.linspace(0,1,samples)
wave = np.sin(w*time)+0.5*np.sin(w*2*time)+0.75*np.sin(w*6*time)
fft1 = fftpack.fft(wave, time.size)
fftabs1 = np.abs(fft1)
xf = fftpack.fftfreq(time.size, time[-1]/samples)#the length in seconds divided by samples
#xf = fftpack.fftfreq(time.size, (len(wave)/samples)/samples) #finding the length in seconds divided by samples
nyq = 0.5*samples
cutoff_freq = 30
cut = cutoff_freq/nyq
b,a = signal.butter(5,cut, btype ='lowpass', analog = False)
yfilt = signal.filtfilt(b,a,wave)

fft2 = fftpack.fft(yfilt, time.size)
fftabs2 = np.abs(fft2)

plt.figure()
plt.subplot(1,3,1)
plt.plot(time, wave)

plt.subplot(1,3,2)
plt.plot(xf[0:len(xf)//2],(2/wave.size)*fftabs1[0:len(fftabs1)//2])

plt.subplot(1,3,3)
plt.plot(xf[0:len(xf)//2],(2/wave.size)*fftabs2[0:len(fftabs2)//2])
plt.show()