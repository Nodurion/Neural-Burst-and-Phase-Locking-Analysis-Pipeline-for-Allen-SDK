import os
import numpy as np
import pandas as pd
import pywt
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import json
import math
import itertools
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

# Set display option to show all columns
pd.set_option('display.max_columns', None)

output_dir = "D:\Lowet_lab\AllenSDK\cache"
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=output_dir)

# Get the metadata tables
units_table = cache.get_unit_table()
channels_table = cache.get_channel_table()
probes_table = cache.get_probe_table()
behavior_sessions_table = cache.get_behavior_session_table()
ecephys_sessions_table = cache.get_ecephys_session_table()

# Variables & Area Combinations
area_names = np.unique(channels_table['structure_acronym'])
area_list = np.array(['CA1', 'CA3', 'DG', 'LGd', 'SUB', 'VISp', 'VISam'], object)
area_combinations = list(itertools.product(area_list, repeat=2))
print(area_combinations)

distance_devision = 200000 
window_before_ms = 1250
window_after_ms = 750
FS = 1250
theta_band = [3, 10]
gamma_band = [30, 60]
bin_width = 1
freq_bins = np.arange(1, 100, bin_width)

def filter_freq_band(signal, FS, band):
    nyquist = 0.5 * FS
    low, high = band[0] / nyquist, band[1] / nyquist
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)

def calculate_mvl(angles_all_trials):
    if len(angles_all_trials) < 10:
        return np.nan
    mean_vector_length = np.abs(np.nanmean(np.exp(1j * np.array(angles_all_trials))))
    if mean_vector_length == 1.0:
        return np.nan
    NT = np.sum(~np.isnan(angles_all_trials))
    MLVsquare = mean_vector_length ** 2
    MLV_unbiased = (1 / (NT - 1)) * ((MLVsquare * NT) - 1)
    return MLV_unbiased

def detect_bursts_per_unit(spike_times_per_unit, burst_time_threshold, min_spikes_in_burst=3):
    print("Detecting bursts per unit")
    burst_spikes_per_unit = {}
    for unit, spike_data in spike_times_per_unit.items():
        num_trials, num_timepoints = spike_data.shape
        burst_spikes_unit = np.zeros_like(spike_data, dtype=np.int8)
        for trial_idx in range(num_trials):
            spiketimes = np.where(spike_data[trial_idx] == 1)[0]
            burst_vector = np.zeros(num_timepoints, dtype=np.int8)
            if len(spiketimes) > 0:
                burst_start = spiketimes[0]
                burst_count = 1
                for spike_time in spiketimes[1:]:
                    if spike_time - burst_start < burst_time_threshold:
                        burst_count += 1
                    else:
                        if burst_count >= min_spikes_in_burst:
                            burst_vector[burst_start] = 1
                        burst_start = spike_time
                        burst_count = 1
                if burst_count >= min_spikes_in_burst:
                    burst_vector[burst_start] = 1
            burst_spikes_unit[trial_idx, :] = burst_vector
        burst_spikes_per_unit[unit] = burst_spikes_unit
    return burst_spikes_per_unit

def detect_single_spikes_per_unit(spike_times_per_unit, burst_time_threshold, min_spikes_in_burst=3):
    print("Detecting single spikes per unit")
    single_spikes_per_unit = {}
    for unit, spike_data in spike_times_per_unit.items():
        num_trials, num_timepoints = spike_data.shape
        single_spikes_unit = np.zeros_like(spike_data, dtype=np.int8)
        for trial_idx in range(num_trials):
            spiketimes = np.where(spike_data[trial_idx] == 1)[0]
            single_spike_vector = np.zeros(num_timepoints, dtype=np.int8)
            burst_start = None
            burst_count = 0
            if len(spiketimes) > 0:
                for i, spike_time in enumerate(spiketimes):
                    if burst_start is None:
                        burst_start = spike_time
                        burst_count = 1
                    elif spike_time - burst_start < burst_time_threshold:
                        burst_count += 1
                    else:
                        if burst_count < min_spikes_in_burst:
                            for j in range(i - burst_count, i):
                                single_spike_vector[spiketimes[j]] = 1
                        burst_start = spike_time
                        burst_count = 1
                if burst_count < min_spikes_in_burst:
                    for j in range(len(spiketimes) - burst_count, len(spiketimes)):
                        single_spike_vector[spiketimes[j]] = 1
            single_spikes_unit[trial_idx, :] = single_spike_vector
        single_spikes_per_unit[unit] = single_spikes_unit
    return single_spikes_per_unit
