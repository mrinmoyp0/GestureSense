# GestureSense


# GestureSense: A Real-Time Hand Gesture Recognition Application




## Overview

This project demonstrates real-time gesture recognition using a webcam, OpenCV, and scikit-learn's Support Vector Machine (SVM) classifier. Hand landmarks are extracted using MediaPipe, and the SVM model is trained to recognize gestures based on these landmarks.
## Introduction

GestureSense uses the MediaPipe library for hand tracking and OpenCV for image processing. The hand landmarks extracted from the frames are used as features for the SVM classifier, trained on a dataset of hand gestures.
## Prerequisites

Before running GestureSense, make sure you have the following installed:

Python (>=3.6),

pip (package installer for Python)
## Installing Libraries

Install the required packages using:

```bash
  pip install numpy opencv-python scikit-learn mediapipe

```
## Installation

1: Clone the repository:

```bash
  git clone https://github.com/mrinmoyp0/GestureSense.git

```
2: Navigate to the project directory:

```bash
  cd GestureSense

```
3: Install the required dependencies:

```bash
  pip install -r requirements.txt

```
## Usage

**Live Hand Gesture Recognition**

1: Run the Streamlit app:

```bash
  streamlit run GestureSense.py

```
2: Choose "Live Hand Gesture Recognition" from the dropdown menu.

3: Click "Start Webcam" to begin real-time gesture recognition.

4: The recognized gesture will be displayed on the webcam feed.

5: Click "Stop Webcam" to end the webcam stream.

**Hand Recognition from Image**

1: Run the Streamlit app:

```bash
  streamlit run GestureSense.py

```
2: Choose "Hand Recognition from Image" from the dropdown menu.

3: Upload an image containing a hand gesture using the file uploader.

4: The recognized gesture will be displayed along with the processed image.
## File Structure

| File / Directory        | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| GestureSense.py         | Main application script for Streamlit UI and gesture recognition. |
| Image_Data/             | Directory containing the dataset of hand gesture images.      |
| Model/                  | Directory to save the trained SVM model and scaler.           |
| Hand_Gestures_Recognisation.ipynb          | This is a complete python script in notebook format (.ipynb), which will run on Jyupeter Notebook and contains Loading, Training and Predicting all of them in a single file.                   |
| Main_HGR.py              | This python script runs directly on any IDE.                                            |
| Run App GestureSense.cmd       | An cmd code for running the app with just a double click.                   |
| Train_SVM_Model.py                 | Python script to Train the SVM model on Image Data                                            |
| animation.json          | JSON file containing Lottie animation data.                   |
| requirements.txt        | List of Python dependencies.                                  |


## Dependencies

- NumPy
- scikit-learn
- OpenCV
- MediaPipe
- Streamlit
- joblib

Install the required packages using:

```bash
  pip install -r requirements.txt

```
