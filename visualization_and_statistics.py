import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import scipy.stats as stats
import json
import itertools
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

bin_width = 1
bins = 72
freq_bins = np.arange(1, 20, bin_width)
theta_lower = 3 
theta_upper = 10

area_list = np.array(['CA1', 'CA3', 'DG', 'LGd', 'SUB', 'VISp', 'VISam'], object)
area_combinations = list(itertools.product(area_list, repeat=2))

def safe_mean_std(data):
    if len(data) == 0:
        return np.nan, np.nan
    mean = np.nanmean(data)
    std_err = np.nanstd(data) / np.sqrt(np.sum(~np.isnan(data)))
    return mean, std_err

def plot_polar_hist(ax, angles, title, color):
    ax.hist(angles, bins=bins, density=True, color=color, alpha=0.7)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    ax.set_title(title)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

# Plotting Frequency Distribution 
def plot_frequency_distribution(freq_bins, mean_spikes, std_spikes, mean_bursts, std_bursts, area):
    plt.figure(figsize=(12, 6))
    plt.plot(freq_bins, np.array(mean_spikes), label='Spikes', color='orange')
    plt.fill_between(freq_bins, np.array(mean_spikes) - np.array(std_spikes),
                     np.array(mean_spikes) + np.array(std_spikes), color='orange', alpha=0.3)
    
    plt.plot(freq_bins, np.array(mean_bursts), label='Bursts', color='purple')
    plt.fill_between(freq_bins, np.array(mean_bursts) - np.array(std_bursts),
                     np.array(mean_bursts) + np.array(std_bursts), color='purple', alpha=0.3)

    plt.axvspan(theta_lower, theta_upper, color='gray', alpha=0.05, hatch='//')
    plt.text((theta_lower + theta_upper) / 2, 0.175 + 0.002,
             f'Theta Range ({theta_lower}-{theta_upper} Hz)', ha='center', color='grey', fontsize=10)
    plt.ylim(0, 0.175)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'$(\mathrm{Phase\ Locking\ Value})^2$')
    plt.title(f'{area} Distribution of Phase Locking Values by Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
