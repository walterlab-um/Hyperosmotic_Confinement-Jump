# HOPS Condensate SPT during Hyperosmotic Stress

This repository contains Python scripts to analyze single molecule tracking (SMT) of HOPS condensates formed by DCP1A during hyperosmotic stress, ER docking analysis, and GEMs accessibility mapping.  
Please consider to cite: "ER tethering and active transport govern condensate diffusion during hyperosmotic stress" (Genome Biology, 2026)

## Installation

All prerequisites to run the scripts in this repository are specified in `conda_environment-spt.yml`  
A `spt` conda environment can be installed from the yml file by:  
```bash
conda env create -f environment.yml
Activate the spt conda environment before running any scripts

bash
conda activate spt
A basic pipeline to analyze diffusion
[optional] If dual-color SPT is needed, channel registration should be performed to align two videos from each channel, and scripts in folder "Camera_Registration" can be used.

Filter out non-single molecule signals in the SMT videos using "bandpass_filter.py"

Extract SMT trajectories using TrackMate (https://imagej.net/plugins/trackmate/)

Export trajectories as csv files using "Export Tracks" or "Export Spots" function in TrackMate

Calculate running-window MSD analysis from every single trajectory using the main processing script. The output includes:

trackID

list of time (s)

list of x positions (µm)

list of y positions (µm)

R2_loglog (log-log MSD fit)

alpha (anomalous diffusion exponent)

D_loglog (log-log diffusion coefficient, µm²/s)

R2_linear (linear MSD fit)

D_linear (linear diffusion coefficient, µm²/s)

Pooling all datasets from different dates of experiments under the same condition using concatenation scripts

Classification of sub/super-diffusion and distribution plots using plotting scripts in "python_codes"

Additional analyses
To analyze ER docking between condensates and ER structures, minimum distance analysis using "calculate_min_distances.py"

License
MIT License (LICENSE file).
