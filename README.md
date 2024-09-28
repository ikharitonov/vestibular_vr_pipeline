# VestibularVR Analysis Pipeline

This is the general pipeline for loading, preprocessing, aligning, quality checking and applying basic analysis to the data recorded on the <a href=https://ranczlab.github.io/RPM/>RPM</a> (e.g. running) using <a href=https://harp-tech.org/index.html>HARP devices</a>, eye movements data derived from <a href=https://sleap.ai/>SLEAP</a> and neural data (fiber photometry, Neuropixels).

## Installation

The code mainly relies on <a href=https://github.com/harp-tech/harp-python>harp-python</a> and <a href=https://github.com/SainsburyWellcomeCentre/aeon_mecha>aeon_mecha</a> packages. The proposed setup is to first create an Anaconda environment for _aeon\_mecha_, install it and then install _harp-python_ inside of this same environment. Optional packages required by some of the example Jupyter notebooks, but not essential for the main pipeline, are cv2, ffmpeg.

### Create anaconda environment

```python
conda create -n aeon
conda activate aeon
```

### Install _aeon\_mecha_

```python
git clone https://github.com/SainsburyWellcomeCentre/aeon_mecha.git
cd aeon_mecha
python -m pip install -e .
```

### Install _harp-python_

```python
pip install harp-python
```

## Repository contents

```
📜demo_pipeline.ipynb
📜grab_figure.ipynb
📂harp_resources
 ┣ 📄utils.py
 ┣ 📄process.py
 ┣ 📄h1-device.yml
 ┗ 📄h2-device.yml
 ┗ 📂notebooks
    ┣ 📜load_example.ipynb
    ┣ 📜demo_synchronisation.ipynb
    ┣ 📜Treshold_exploration_Hilde.ipynb
    ┣ 📜comparing_clocked_nonclocked_data.ipynb
    ┗ 📜prepare_playback_file.ipynb
📂sleap
 ┣ 📄load_and_process.py   -->   main functions for SLEAP preprocessing pipeline
 ┣ 📄add_avi_visuals.py   -->   overlaying SLEAP points on top of the video and saving as a new one for visual inspection
 ┣ 📄horizontal_flip_script.py   -->   flipping avi videos horizontally using OpenCV
 ┣ 📄registration.py   -->   attempt at applying registration from CaImAn to get rid of motion artifacts (https://github.com/flatironinstitute/CaImAn/blob/main/demos/notebooks/demo_multisession_registration.ipynb)
 ┣ 📄upscaling.py   -->   attempt at applying LANCZOS upsampling to avi videos using OpenCV to minimise SLEAP jitter
 ┗ 📂notebooks
    ┣ 📜batch_analysis.ipynb
    ┣ 📜ellipse_analysis.ipynb   -->   visualising SLEAP preprocessing outputs
    ┣ 📜jitter.ipynb   -->   quantifying jitter inherent to SLEAP
    ┣ 📜light_reflection_motion_correction.ipynb   -->   segmentation of light reflection in the eye using OpenCV (unused)
    ┣ 📜saccades_analysis.ipynb   -->   step by step SLEAP data preprocessing (now inside of load_and_process.py + initial saccade detection
    ┗ 📜upsampling_jitter_analysis.ipynb   -->   loading SLEAP outputs from LANCZOS upsampling tests
```

## Conventions

SLEAP outputs to be saved as VideoData2_...sleap.csv
Flipped videos to be saved as VideoData2_...flipped.avi

## Functions available

### HARP

### SLEAP
