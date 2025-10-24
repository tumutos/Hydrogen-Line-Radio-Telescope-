#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 23:44:42 2025

@author: umutcansu
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from python_hackrf import pyhackrf
import time
import scipy.signal as scp
from tqdm.auto import tqdm
import csv
#%%

#########################################
########### PARAMETERS ##################
#########################################
center_freq = 96.6e6#OL frequency 500KHz ofsset  HL: 1420.9058e6  bilkent_fm: 96.6e6
freq_offset = int(500e3)
sample_rate = 5e6
fft_size = 4096
recording_time =300 #420 sec 7 minutes
baseband_filter = 1e6
lna_gain = 16 # 0 to 40 dB in 8 dB steps
vga_gain = 20 # 0 to 62 dB in 2 dB steps
averaging_coef = 2e16 

#########################################
################# main ##################
#########################################
pyhackrf.pyhackrf_init()
sdr = pyhackrf.pyhackrf_open()
allowed_baseband_filter = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(baseband_filter) # calculate the supported bandwidth relative to the desired one

sdr.pyhackrf_set_sample_rate(sample_rate)
sdr.pyhackrf_set_baseband_filter_bandwidth(allowed_baseband_filter)
sdr.pyhackrf_set_antenna_enable(False)  # It seems this setting enables or disables power supply to the antenna port. False by default. the firmware auto-disables this after returning to IDLE mode

sdr.pyhackrf_set_freq(center_freq)
sdr.pyhackrf_set_amp_enable(False)  # False by default
sdr.pyhackrf_set_lna_gain(lna_gain)  # LNA gain - 0 to 40 dB in 8 dB steps
sdr.pyhackrf_set_vga_gain(vga_gain)  # VGA gain - 0 to 62 dB in 2 dB steps

print(f'center_freq: {center_freq} sample_rate: {sample_rate} baseband_filter: {allowed_baseband_filter}')
num_samples = int(recording_time * sample_rate)
samples = np.zeros(num_samples, dtype=np.complex64)
last_idx = 0

def rx_callback(device, buffer, buffer_length, valid_length):  # this callback function always needs to have these four args
    global samples, last_idx

    accepted = valid_length // 2
    accepted_samples = buffer[:valid_length].astype(np.int8) # -128 to 127
    accepted_samples = accepted_samples[0::2] + 1j * accepted_samples[1::2]  # Convert to complex type (de-interleave the IQ)
    accepted_samples /= 128 # -1 to +1
    samples[last_idx: last_idx + accepted] = accepted_samples

    last_idx += accepted

    return 0

sdr.set_rx_callback(rx_callback)
sdr.pyhackrf_start_rx()
print('is_streaming', sdr.pyhackrf_is_streaming())

total_steps = 10 * recording_time
with tqdm(total=total_steps, desc="Observing…", leave=False, ncols=75) as pbar:
    for _ in range(total_steps):
        time.sleep(0.1)
        pbar.update(1)
print("Observation Completed...")
sdr.pyhackrf_stop_rx()
sdr.pyhackrf_close()
pyhackrf.pyhackrf_exit()
#num_rows = len(samples)//fft_size
#samples = np.reshape(samples[0:fft_size*num_rows],(-1,4096)) # rehsape the samples in the sahep (a , 4096)


#########################################
############### Filtering ###############
#########################################
print("Filtering...")
t = np.arange(num_samples) / sample_rate
dc_shift = np.exp(-2j * np.pi * freq_offset * t)
baseband = samples * dc_shift

cut_off = int(200e3)
numtaps = 257
nyq = 0.5 * sample_rate
taps = scipy.signal.firwin(numtaps , cut_off, window='hamming', fs = sample_rate)
filtered = scipy.signal.lfilter(taps, 1.0, baseband)


#########################################
################# FFT ###################
#########################################
# 50% overlap framing
hop = fft_size // 2
num_frames = (len(filtered) - fft_size) // hop + 1
print("Taking FFT...")
frames = np.lib.stride_tricks.as_strided(
    filtered,
    shape=(num_frames, fft_size),
    strides=(filtered.strides[0]*hop, filtered.strides[0])
)

# apply window, FFT, magnitude-squared
window = np.hamming(fft_size)
spec   = np.fft.fftshift(np.fft.fft(frames * window[np.newaxis, :], axis=1)) #faz kontorlü yap ona göre mean al!
power  = np.abs(spec)**2

#########################################
############# POST-PROCESS ##############
#########################################

mean = np.mean(power, axis = 0)
np.save("on_source_10.npy",np.array(mean))

freqs = np.linspace(center_freq - sample_rate/2, center_freq+sample_rate/2, fft_size, endpoint=False)
mask = (freqs < center_freq-100e3) | (freqs > center_freq+100e3) 
coeffs = np.polyfit(freqs[mask], mean[mask],3)
baseline = np.polyval(coeffs, freqs)
mean = mean - baseline

#savitzky-golay filtering
smoothed = scp.savgol_filter(mean,window_length=21,polyorder=3,mode = 'mirror')


mag = 10*np.log10(smoothed.clip(min=1e-12))
#back_ground = 10*np.log10(smoothed.clip(min=1e-12))


#%%
#########################################
############## VISUALIZATION ############
#########################################


# waterfall of power vs time
print("Drawing the graph...")
# plt.figure(1,figsize=(8,6)).clf()
# plt.imshow(
#     10*np.log10(power.T.clip(min = 1e-12)),
#     aspect='auto',
#     origin='lower',
#     extent=[0, recording_time, center_freq - sample_rate/2, sample_rate/2 + center_freq]
# )
# plt.colorbar(label="Power (dB)")
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.title("H-line Preprocessed Spectrum")
# plt.show()
#%%
ax = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/sample_rate)) + center_freq
plt.figure(2).clf()
plt.plot(ax,mag)
plt.title("Magnitude of the Signal")
plt.xlabel("Frequency")
plt.ylabel("dB")



# on_source = np.load("on_source.npy")
# off_source = np.load("back_g.npy")
# diff = on_source - off_source
# #diff = 10*np.log10(diff.clip(min = 1e-12))


# plt.figure(3).clf()
# plt.plot(ax , diff)






