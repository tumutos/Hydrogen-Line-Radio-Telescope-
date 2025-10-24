#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:15:10 2025

@author: umutcansu
"""

import numpy as np 
import scipy.signal as scp
import matplotlib.pyplot as plt
import pandas
from scipy.ndimage import gaussian_filter1d  
from astropy.coordinates import SkyCoord
import astropy.units as u
#%%

f_cen = 1420.4058e6
f_samp = 5e6
N = 4096



dir_data = "/Users/umutcansu/Desktop/Radyo_teleskop/observation_data/d200525"
min_length = float("inf")
#%%
for i in range(1,13):
    dir_name = dir_data + f"/onsource_{i*5}.dat"
    data = np.fromfile(dir_name, dtype=np.float32)
    size = len(data)
    if min_length > size:
        min_length = size
    


#%% 
dir_offsource = "/Users/umutcansu/Desktop/Radyo_teleskop/observation_data/d200525/offsource.dat"
back_g = np.fromfile(dir_offsource, dtype=np.float32)
size_b = len(back_g) // 4096
back_g = back_g[0:4096*size_b]
back_g = np.reshape(back_g,[-1,4096])
mean_g = np.mean(back_g,axis = 0)


X = np.zeros([12,4096],dtype = np.float32) 
for i in range(1,13):
    print(f"Looking into observation {i}/12")
    dir_name = dir_data + f"/onsource_{i*5}.dat"
    data = np.memmap(dir_name, dtype=np.float32,mode ='r')
    data = data[0:4096*size_b]
    data = np.reshape(data, [-1, 4096])
    mean = np.mean(data,axis=0,dtype=np.float32)
    X[i-1,:] = mean
    
    



#%%
data_mean = X[2:6,:]
data_mean = np.mean(data_mean,axis = 0)

smooth_justdata = scp.savgol_filter(data_mean,window_length=31,polyorder=3,mode = 'mirror')
smooth_noise = scp.savgol_filter(mean_g,window_length=31,polyorder=3,mode = 'mirror')
#data_mean = np.mean(X,axis = 0,dtype=np.float32)
smooth_dif = smooth_justdata -smooth_noise
diff = data_mean  - mean_g
freqs = np.linspace(f_cen - f_samp/2, f_cen+f_samp/2, N, endpoint=False)
toggle_mask = (freqs > f_cen-500e3) & (freqs < f_cen+500e3) 
mask = (freqs < f_cen-500e3) | (freqs >f_cen+500e3) 
coeffs = np.polyfit(freqs[mask], diff[mask],3)
baseline = np.polyval(coeffs, freqs)
coarse_d = diff - baseline
smoothed =scp.savgol_filter(coarse_d,window_length=31,polyorder=3,mode = 'mirror')
smoothed2 =scp.savgol_filter(diff[toggle_mask],window_length=31,polyorder=3,mode = 'mirror')
mag = 10*np.log10(smoothed.clip(min = 1e-12))
mag_data = 10*np.log10(smooth_justdata)
mag_backg = 10*np.log10(mean_g)


#%% 
#Here i ll substract the min from both mean data and mean back ground noise 

s_a = smooth_justdata
s_b = smooth_noise
transpose = np.matmul(s_a , np.transpose(s_b))
mult = transpose*s_b
s_ahat = (mult/(np.linalg.vector_norm(s_b)*np.linalg.vector_norm(s_b)))
s_be = s_a - s_ahat
s_be = scp.savgol_filter(s_be,window_length=31,polyorder=3,mode = 'mirror')


plt.figure(7).clf()
plt.subplot(4,1,1)
plt.plot(freqs,s_be)
plt.title("S_be")
plt.subplot(4,1,2)
plt.plot(freqs,s_a)
plt.title("S_a")
plt.subplot(4,1,3)
plt.plot(freqs,s_ahat)
plt.title("S_ahat")
plt.subplot(4,1,4)
plt.plot(freqs,s_b)
plt.title("S_b")

plt.figure(8).clf()
plt.plot(freqs,s_be)
plt.title("Plot of Difference")
plt.xlabel("Frequency")
plt.ylabel("Power(cts)")

#%%



data_slice = data_mean[toggle_mask]
data_slice = scp.savgol_filter(data_slice,window_length=31,polyorder=3,mode = 'mirror')
back_slice = mean_g[toggle_mask]
diff_slice = data_slice - back_slice

plt.figure(1).clf()
plt.subplot(3,1,1)
plt.plot(freqs,smoothed)
plt.title("Baseline subs, smoothing after diff")
plt.subplot(3,1,2)
plt.plot(freqs[toggle_mask],smoothed2)
plt.title("NO Baseline subs, smoothing after diff")
plt.subplot(3,1,3)
plt.plot(freqs,smooth_dif)
plt.title("NO Baseline subs, smoothing before diff")



plt.figure(2).clf()
plt.plot(freqs, mag_data)
plt.axvline(f_cen)
plt.axvline(f_cen - 500e3)
plt.axvline(f_cen + 500e3)
plt.title("onsource")

plt.figure(3).clf()
plt.plot(freqs, mean_g)
plt.title("back ground")

plt.figure(4).clf()
plt.subplot(311)
plt.plot(freqs[toggle_mask],10*np.log10(data_slice))
plt.axvline(f_cen)
plt.axvline(f_cen - 100e3)
plt.axvline(f_cen + 100e3)
plt.subplot(312)
plt.plot(freqs[toggle_mask],back_slice)
plt.axvline(f_cen)
plt.subplot(313)
plt.plot(freqs[toggle_mask],diff_slice)
plt.axvline(f_cen)

plt.figure(5).clf()
plt.plot(freqs,baseline)

#%%


# freq_hz    : 1D array of frequency bins [Hz]
# spec_diff  : on–off difference **linear** power spectrum

f0 = 1_420_405_800.0   # rest freq of H I [Hz]

# 1) local‐arm component: within ±100 kHz of f0
mask1 = np.abs(freqs - f0) < 100e3
f_peak1 = freqs[mask1][np.argmax(s_be[mask1])]

# 2) distant‐arm component: between +100 kHz and +300 kHz
mask2 = (freqs - f0 > 100e3) & (freqs - f0 < 300e3)
f_peak2 = freqs[mask2][np.argmax(s_be[mask2])]

print(f"Local‐arm peak:   {f_peak1/1e6:.6f} MHz")
print(f"Distant‐arm peak: {f_peak2/1e6:.6f} MHz")

c    = 299_792.458   # km/s
v1   =  c*(f0 - f_peak1)/f0
v2   =  c*(f0 - f_peak2)/f0
print(f"V_local ≃ {v1:.1f} km/s,  V_perseus ≃ {v2:.1f} km/s")


#%%



sky = SkyCoord("19h00m20.02s", "18d36m03.6s", frame="icrs")

# convert
l = sky.galactic.l.deg
b = sky.galactic.b.deg
print(f"Galactic (l, b) = ({l:.2f}°, {b:.2f}°)")



V0 = 220.0                   # km/s
l_rad = np.deg2rad(l)
b_rad = np.deg2rad(b)

v_pred = V0 * np.sin(l_rad) * np.cos(b_rad)
print(f"Expected V_LSR ≃ {v_pred:.1f} km/s")

#%%


import numpy as np
import matplotlib.pyplot as plt


vel, Tb = np.loadtxt('spectrum.txt', skiprows=4, usecols=(0,1), unpack=True)



v_local  = -32.967 # km/s
v_perseus = -74.647 # km/s

# Plot
plt.figure(figsize=(8,4))
plt.plot(vel, Tb, lw=1.2, label='LAB T₍b₎')
plt.axvline(v_local,  ls='--', color='C1', label=f'Local arm ({v_local} km/s)')
plt.axvline(v_perseus, ls='--', color='C2', label=f'Perseus arm ({v_perseus} km/s)')

plt.xlim(-150, 150)
plt.xlabel('V$_{LSR}$ (km/s)')
plt.ylabel('Brightness Temperature (K)')
plt.title('LAB 21 cm profile at ℓ=50.54°, b=6.47°')
plt.legend()
plt.tight_layout()
plt.show()




