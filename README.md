# MiDaS Live Depth Estimation using OpenCV and Torch libraries

This repository contains a Python script for performing live depth estimation using the MiDaS model. The script captures live video from a camera and applies the MiDaS model to generate depth maps in real-time.

## Requirements
Before running the script, ensure you have the following dependencies installed:
- OpenCV
- PyTorch
- Matplotlib
- Numpy

You can install these dependencies using pip:

```bash
pip install opencv-python torch matplotlib numpy
```

## Usage

### Running the Script

To run the live depth estimation script, simply execute the following command:

```bash
python depth_estimation.py
```

### Selecting the Model Type

The script supports three different model types for depth estimation:

- DPT_LARGE
- DPT_Hybrid
- MIDAS_SMALL

You can specify the model type by modifying the run function call at the end of the script:

```run(ModelType.DPT_LARGE)```

## Figure

![MiDaS Inference Screenshot](https://github.com/user-attachments/assets/8594d07f-ac01-4c27-84e9-7d9544fc8e8b)
