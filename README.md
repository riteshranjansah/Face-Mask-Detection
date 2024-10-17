# Face Mask Detection

This repository provides a comprehensive solution for detecting whether a person is wearing a mask or not. The model is built using TensorFlow, Keras, and OpenCV, and it employs a pre-trained MobileNetV2 model for classification. The solution includes scripts to train the model, as well as to run real-time detection using a webcam.

## Repository Structure

### 1. **dataset/**: 
Contains the dataset for training the model:
- **with_mask/**: Images of people wearing a mask.
- **without_mask/**: Images of people not wearing a mask.

### 2. **face_detector/**: 
Contains a pre-trained face detection model:
- **deploy.prototxt**: The configuration file for the face detector.
- **res10_300x300_ssd_iter_140000.caffemodel**: Pre-trained weights for the face detector.

### 3. **files/**: 
- **mask_detector.model**: The trained mask detection model.
- **plot.png**: Plot of the model's training loss and accuracy.

### 4. **Face-Mask-Detection/**: 
Contains the Python scripts for training and real-time detection.

- **train_mask_detection.py**: Script for training the face mask detection model.
- **detect_mask.py**: Script to detect face masks in real-time using the webcam.

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- TensorFlow 2.x
- OpenCV
- imutils
- scikit-learn
- matplotlib

You can install the required Python packages using the following command:

```bash
pip install tensorflow opencv-python imutils scikit-learn matplotlib
```

## Usage

### 1. Training the Model

To train the mask detection model, run the following command:

```bash
python Face-Mask-Detection/train_mask_detection.py
```

This script will:
- Load images from the **dataset/** folder.
- Preprocess the images and train a MobileNetV2 model.
- Save the trained model as **mask_detector.model** and a training plot as **plot.png**.

### 2. Detecting Face Masks in Real-Time

To run real-time mask detection using your webcam, run the following command:

```bash
python Face-Mask-Detection/detect_mask.py
```

This script will:
- Load the face detector model and the trained mask detector model.
- Continuously capture frames from the webcam.
- Detect faces and predict whether the face is wearing a mask or not.
- Display the result in real-time on the webcam feed.

To exit the real-time detection, press **q** in the terminal where the script is running.

## How It Works

1. **Face Detection**: The **detect_mask.py** script uses OpenCV’s DNN module to load a pre-trained face detection model (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`). The face detector identifies faces in each video frame.
  
2. **Mask Prediction**: Once faces are detected, the face images are cropped and resized to 224x224 pixels, then passed through the trained mask detection model (`mask_detector.model`). The model predicts whether the person is wearing a mask or not.

3. **Results**: The bounding box and the prediction (i.e., "Mask" or "No Mask") are displayed on the video feed in real-time.

## File Descriptions

### 1. **train_mask_detection.py**

This script trains the face mask detection model using the MobileNetV2 architecture. The model uses the **with_mask** and **without_mask** dataset categories, and applies data augmentation techniques for better generalization.

### 2. **detect_mask.py**

This script is for real-time face mask detection. It uses OpenCV to open the webcam feed, detects faces using a pre-trained face detector, and predicts if the detected person is wearing a mask using the trained model.

### 3. **mask_detector.model**

This is the trained model that can predict whether a person is wearing a mask or not.

### 4. **plot.png**

A plot showing the training loss and accuracy of the model during training.

## License

This project is open-source and available for personal use and modification. No formal license has been applied, so feel free to use or modify the code as per your needs.

## Acknowledgments

- The face detector model is based on a pre-trained OpenCV DNN model (`res10_300x300_ssd_iter_140000.caffemodel`).
- The MobileNetV2 architecture was originally developed by Google and is available in the Keras library.
