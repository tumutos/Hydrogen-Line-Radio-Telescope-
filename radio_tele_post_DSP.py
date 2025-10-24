#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 23:09:19 2025

@author: umutcansu
"""

import numpy as np 
import scipy.signal as scp
import matplotlib.pyplot as plt
import pandas
from scipy.ndimage import gaussian_filter1d  

dir_data = "/Users/umutcansu/Desktop/Radyo_teleskop/observation_data/d080525/test3.dat"
dir_background = "/Users/umutcansu/Desktop/Radyo_teleskop/observation_data/d080525/test2.dat"

data = np.memmap(dir_data, dtype=np.float32,mode ='r')
back_g = np.fromfile(dir_background, dtype=np.float32)

spec_data = len(data)//4096
spec_noise = len(back_g)//4096
array = np.array([spec_data,spec_noise])
spec = np.min(array)
data_r = np.reshape(data[(spec-500)*4096:spec*4096],(-1,4096))

bck_g_r = np.reshape(back_g[0:spec*4096],(-1,4096))



#%%
N = 4096
fs = 2.048e6
center = 1.4204058e9
f_cen = 1420.4058e6
f_samp = 2.048e6 
frames_per_chunk = 388450 #5minutes worth of data


dat = data_r[400,:]
bck = bck_g_r[0,:]
dat_2 = data_r[1,:]
mean_s = np.mean(data_r,axis=0,dtype = np.float32)
mean_s_bck = np.mean(bck_g_r,axis=0,dtype = np.float32)

mag = 10*np.log10(dat,dtype = np.float32)
mag_avg = 10*np.log10(mean_s,dtype = np.float32)
mag_bck = 10*np.log10(bck,dtype = np.float32)
mag_avg_bck = 10*np.log10(mean_s_bck,dtype = np.float32)
#mag_n =np.log10(np.abs(dat - bck)) #dB of one sample (noisless)


diff = mean_s - mean_s_bck

# freqs = np.linspace(f_cen - f_samp/2, f_cen+f_samp/2, N, endpoint=False)
# mask = (freqs < center-100e3) | (freqs > center+100e3) 
# coeffs = np.polyfit(freqs[mask], diff[mask],3)
# baseline = np.polyval(coeffs, freqs)

# diff = diff - baseline



#moving square smoothing 
# window = np.ones(15) /15
# smoothed = np.convolve(diff_b, window, mode='same')


#savitzky-golay filtering
#smoothed = scp.savgol_filter(diff,window_length=21,polyorder=3,mode = 'mirror')


#gaussian smoothing
smoothed = gaussian_filter1d(diff, sigma=2) 

 




mag_avg_n = 10*np.log10(smoothed.clip(min = 1e-12),dtype = np.float32) #dB of the mean (noisles)
mag_avg_n_b = 10*np.log10(diff.clip(min=1e-12),dtype = np.float32)
bin_freq = 500 #500 Hz is the frequency per bin

#%%

ax = np.linspace(f_cen - f_samp/2, f_cen+f_samp/2,N,endpoint=False )
plt.figure(1).clf()
#plt.subplot(211)
plt.plot( ax,mag_avg)
plt.xlabel("Frequency")
plt.title(" Average Graph ")
# plt.subplot(212)
# plt.plot(ax[524:1524],mag_avg[524:1524])
# plt.title("Averaged Graph  ")
# plt.xlabel("Frequency")


plt.figure(2).clf()
#plt.subplot(211)
plt.plot(ax,mag_avg_bck)
plt.title("Before Average Graph (Back G.) ")
plt.xlabel("Frequency")
# plt.subplot(212)
# plt.plot(ax[524:1524],mag_avg_bck[524:1524],label = "back ground")
# plt.plot(ax[524:1524],mag_avg[524:1524],'r',label = "on object")
# plt.title("Averaged Graph(Back G.) ")
# plt.xlabel("Frequency")


plt.figure(3).clf()
plt.subplot(211)
plt.plot(ax,mag_avg_n,'r')
plt.axvline(f_cen,color = 'b',linestyle = ':', linewidth = 1)
plt.axvline(f_cen + 100e3 ,color = 'g',linestyle = ':', linewidth = 1,label = '1420.5058MHz')
plt.axvline(f_cen - 100e3 ,color = 'g',linestyle = ':', linewidth = 1,label = '1420.3058MHz')
plt.title(" Averaged Graph Noise purrified (smoothed)")
plt.ylabel("dB")
plt.xlabel("Frequency")
plt.subplot(212)
plt.plot(ax,mag_avg_n_b,'r')
plt.title(" Averaged Graph Noise purrified ")
plt.ylabel("dB")
plt.xlabel("Frequency")


plt.figure(4).clf()
plt.plot(mag)
plt.figure(5).clf()
plt.plot(smoothed)

#%%
num = np.shape(data_r)[0]//1000
X = np.zeros((num,4096))
for i in range(num):
    X[i,:] = np.mean(data_r[i*num:(i+1)*num,:],axis = 0)
    
X = 20*np.log10(X)   
    
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(
        X,
        origin='lower',       # earliest sweep at the bottom
        aspect='auto',        # stretch so each pixel = one sample
        interpolation='nearest',
        cmap='viridis',       # perceptually uniform
)
ax.set_xlabel('Frequency [MHz]')
ax.set_ylabel('Sweep index (≈ time)')
ax.set_title('Hydrogen‑line waterfall')
plt.colorbar(im, label='Power [dB]')
plt.show()
    







    