#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 11:40:45 2025

@author: umutcansu
"""

import numpy as np 
import scipy.signal as scp
import matplotlib.pyplot as plt
import pandas
from scipy.ndimage import gaussian_filter1d  


N = 4096
fs = 2.048e6
center = 1.4204058e9
f_cen = 1420.4058e6
f_samp = 2.048e6 
frames_per_chunk =  N * 100000


dir_data = "/Users/umutcansu/Desktop/Radyo_teleskop/observation_data/d080525/test3.dat"

data = np.memmap(dir_data, dtype=np.float32,mode ='r')
onsource = data[0:frames_per_chunk]
offsource = data[frames_per_chunk*7:frames_per_chunk*8]
shape_data = len(onsource)//N

slice_on = np.reshape(onsource[0:N*shape_data],(-1,N))
slice_off = np.reshape(offsource[0:N*shape_data],(-1,N))
mean_on = np.mean(slice_on,axis = 0)
mean_off = np.mean(slice_off,axis=0)

#%%
#savitzky-golay filtering
smoothed_on = scp.savgol_filter(mean_on,window_length=43,polyorder=3,mode = 'mirror')
smoothed_off = scp.savgol_filter(mean_off,window_length=43,polyorder=3,mode = 'mirror')


# smoothed_on = gaussian_filter1d(mean_on, sigma=2) 
# smoothed_off = gaussian_filter1d(mean_off, sigma=2) 


diff = smoothed_on - smoothed_off
signal_db_on = 10*np.log10(smoothed_on)
signal_db_off = 10*np.log10(smoothed_off)
diff_db = 10*np.log10(diff.clip(min = 1e-12))

plt.figure(1).clf()
plt.plot(signal_db_on)
plt.title("On source")

plt.figure(2).clf()
plt.plot(signal_db_off)
plt.title("Off source")

plt.figure(3).clf()
plt.plot(diff_db)
plt.title("Difference")
