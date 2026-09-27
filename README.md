# License Plate Recognition and Character Recognition

*A deep learning-based license plate recognition system using TensorFlow and OpenCV.*

## Table of Contents
- [Introduction](#introduction)
- [Project Demo](#project-demo)
- [Patent Application](#patent-application)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)

---

##  Project Demo
https://github.com/user-attachments/assets/7667cb36-da7f-4ae7-8f50-5533d5f9d110


**[Open Front Page UI Video](https://github.com/Pranavkr10/LPR/blob/main/front_page.mp4)**
---

## Patent Application

This project is associated with an Indian patent application titled:

### **An AI-Driven Automatic Number Plate Recognition System**

| Patent Detail | Information |
|---|---|
| **Application No.** | **202511103785 A** |
| **Date of Filing** | 28/10/2025 |
| **Publication Date** | 12/12/2025 |
| **Title** | An AI-Driven Automatic Number Plate Recognition System |
| **Applicant** | Manipal University Jaipur |
| **Country** | India |
| **Inventors** | Mr. Pranav Kumar, Dr. Bhawana Sharma, Dr. Lokesh Sharma |
| **International Classification** | G06T0005400000, G08G0001017000, G06V0020620000, G06V0030146000, G06V0010750000 |

###  Inventors

1. **Mr. Pranav Kumar**
2. **Dr. Bhawana Sharma**
3. **Dr. Lokesh Sharma**

### Applicant

**Manipal University Jaipur**
Off Jaipur-Ajmer Expressway, Post: Dehmi Kalan, Jaipur-303007, Rajasthan, India

### Patent Abstract

The present invention relates to an AI-driven Automatic Number Plate Recognition (ANPR) system. The system comprises an image acquisition module that captures vehicle images from camera feeds or uploaded sources, followed by a multistage image preprocessing pipeline incorporating noise reduction, normalization, Contrast Limited Adaptive Histogram Equalization (CLAHE), and adaptive thresholding to enhance plate visibility in diverse lighting and noise conditions.

A trained model localizes the license plate and recognises through a dual-model CNN architecture for general classification and another for resolving ambiguous character pairs such as **'O' and '0'**. Post-processing with OCR correction and pattern matching validates results against regional plate formats.

The recognized number is cross-checked with a connected vehicle database containing registration details and status flags. A web-based module displays vehicle information and triggers visual or audio alerts for flagged vehicles, enabling efficient real-time verification for law enforcement and smart city surveillance systems.

---

## Introduction

This project detects and recognizes license plate characters using a CNN-based model trained on a dataset of images. The process involves:

**License Plate Localization** – Detect license plate regions.
**Character Segmentation** – Extract individual characters.
**Character Recognition** – Classify characters using a trained neural network.

## An overview of the plate localization

NOTE: This visualization only includes the major steps involved for the detection of a license plate; the actual code includes many other functions to filter out
      the region of interest.

![Plate Localization](https://raw.githubusercontent.com/Pranavkr10/LPR/b1374a44ee4440ecffe0de70b2be97336d037ebb/basic%20steps%20perfomred%20during%20plate%20localization.png)

### Key Features
✔️ Data augmentation for improved robustness.
✔️ Custom F1 score metric for evaluation.
✔️ Uses TensorFlow and Keras for training.
✔️ Processes images to recognize characters from license plates.

---

## Prerequisites
Ensure you have the following installed:

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Google Colab (optional)
- Google Drive (optional for dataset storage)

## Setup and Installation

Install dependencies:

```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib
```

If using Google Colab, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Place your dataset in `/content/drive/MyDrive/info/data`.

## Data Preparation
Dataset structure:

```
/content/drive/MyDrive/info/data/
    ├── train/
    │   ├── class_0/
    │   ├── class_1/
    │   └── ...
    ├── val/
    │   ├── class_0/
    │   ├── class_1/
    │   └── ...
```

 **Training Data:** Used for model learning.
 **Validation Data:** Used for evaluation.

Data augmentation is applied using `ImageDataGenerator`.

## Model Architecture
The CNN model includes:

🔹 **Conv2D Layers** – Feature extraction
🔹 **MaxPooling Layers** – Downsampling
🔹 **Flatten Layer** – Converts feature maps into 1D
🔹 **Dense Layers** – Fully connected for classification

**Model Summary:**

| Layer Type           | Output Shape        | Parameters |
|----------------------|--------------------|------------|
| Conv2D              | (None, 28, 28, 32) | 896        |
| MaxPooling2D        | (None, 14, 14, 32) | 0          |
| Conv2D              | (None, 14, 14, 64) | 18,496     |
| MaxPooling2D        | (None, 7, 7, 64)   | 0          |
| Conv2D              | (None, 7, 7, 128)  | 73,856     |
| MaxPooling2D        | (None, 3, 3, 128)  | 0          |
| Flatten             | (None, 1152)       | 0          |
| Dense               | (None, 256)        | 295,168    |
| Dropout             | (None, 256)        | 0          |
| Dense               | (None, 36)         | 9,252      |

**Total Parameters:** 397,668 (1.52 MB)
**Trainable Parameters:** 397,668
**Non-trainable Parameters:** 0

## Training the Model

**Training settings:**
- **Loss function:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam (`lr=0.001`)
- **Metrics:** Accuracy & custom F1 score
- **Callbacks:** Early stopping & checkpointing

```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=25,
    verbose=1,
    callbacks=callbacks
)
```

### Custom F1 Score Metric
```python
class F1Score(tf.keras.metrics.Metric):
    # Implementation of custom F1 score metric
```

### Early Stopping Callback
Stops training if `val_f1_score` > 99%:
```python
class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs and logs.get("val_f1_score") > 0.99:
            print("\nReached 99% Validation F1 Score, stopping training!!")
            self.model.stop_training = True
```

## Testing the Model
Load and test the model:
```python
model = tf.keras.models.load_model(
    '/content/drive/MyDrive/char_recog1.keras',
    custom_objects={'F1Score': F1Score}  # Register custom metric
)
```

Use `plateLocalization()` to detect plates:
```python
def plateLocalization(imgPath):
    # Image preprocessing, plate localization, and character recognition
    pass
```
---
