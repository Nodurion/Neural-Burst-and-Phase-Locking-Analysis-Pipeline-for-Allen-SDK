# Neural Burst and Phase-Locking Analysis Pipeline for Allen SDK

This repository contains the custom data analysis pipeline built to process Neuropixel datasets from the Allen Brain Observatory Visual Behavior Neuropixels dataset (via AllenSDK). 

It was developed during a Master's research internship at Erasmus Medical Center (2024) to decode how spike bursts and single spikes uniquely phase-lock to network oscillations across distinct brain regions (CA1, CA3, DG, SUB, LGN, VISp, VISam).

## Files included:
* `data_processing.py`: Handles raw data extraction, spike sorting, burst vs. single spike separation using an Inter-Spike Interval (ISI) threshold, and phase-locking value (PLV) calculations relative to local field potentials (LFP).
* `visualization_and_statistics.py`: Generates polar histograms for angular phase distributions, frequency distribution plots, and computes the accompanying circular statistics.

Note: This pipeline utilizes the allensdk to fetch raw Neurodata Without Borders (NWB) files directly from the Allen Brain Observatory S3 cache. Output paths in the scripts reflect the local directory structure used during the internship.
