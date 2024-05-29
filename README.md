
# Drone detection
This is my master's thesis project where I train two neural networks on the Coral Dev Board to investigate whether a faster response (lower latency) or higher precision is more advantageous in a custom-built system for real-time drone tracking with laser. 
## Contents
[Tensorflow 1.15.ipynb](https://github.com/dbongowski/drone-detection/blob/main/Tensorflow%201.15.ipynb "Tensorflow 1.15.ipynb") - is a Jupyter all in one notebook written on Ubuntu 18.04. It allows to install and use Tensorflow 1.15 API with various learning scripts, export model to TFlite compatible with EdgeTPU, manage dataset, generate TFRecords, use LabelImg and create augumentations.
## 

[tflite_model_maker.ipynb](https://github.com/dbongowski/drone-detection/blob/main/tflite_model_maker.ipynb "tflite_model_maker.ipynb") -  is a Jupyter notebook created for transfer learning models on Windows 10 with tflite_model_maker library based on Tensorflow 2. It allows to install and use dependecies, as well as export trained models. 
##

[main.py](https://github.com/dbongowski/drone-detection/blob/main/main.py "main.py") - is a Python program designed for the Coral Dev Board. It features a real-time interpreter for object detection. The code captures the freshest possible frame using OpenCV, processes it with a neural network, filters the neural network's output with a Kalman filter, and calculates the angle from the center to the object using a polynomial function to correct fisheye lens distortions. Data for a 2-axis rotating turret is then sent in int16 format via SPI to an STM32 that controls the laser tracking device.

## Co-author of thesis
-   **[Tomasz Downar-Zapolski](https://github.com/engrPharmacist)** - with high contribution in creating the laser tracking device, author of control system on STM32. Present in every aspect of the project. 
