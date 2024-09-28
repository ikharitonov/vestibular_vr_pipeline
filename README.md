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
 ┣ 📄load_and_process.py
 ┣ 📄add_avi_visuals.py
 ┣ 📄horizontal_flip_script.py
 ┣ 📄registration.py
 ┣ 📄upscaling.py
 ┗ 📂notebooks
    ┣ 📜batch_analysis.ipynb
    ┣ 📜ellipse_analysis.ipynb
    ┣ 📜jitter.ipynb
    ┣ 📜light_reflection_motion_correction.ipynb
    ┣ 📜saccades_analysis.ipynb
    ┗ 📜upsampling_jitter_analysis.ipynb
```

## Functions available

### HARP

### SLEAP
