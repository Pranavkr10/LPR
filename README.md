# License Plate Recognition and Character Recognition

 *A deep learning-based license plate recognition system using TensorFlow and OpenCV.*

## 📌 Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)

##  Introduction
This project detects and recognizes license plate characters using a CNN-based model trained on a dataset of images. The process involves:

**License Plate Localization** – Detect license plate regions.  
**Character Segmentation** – Extract individual characters.  
 **Character Recognition** – Classify characters using a trained neural network.  

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

## 📂 Data Preparation
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

📌 **Training Data:** Used for model learning.  
📌 **Validation Data:** Used for evaluation.  

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

###  Custom F1 Score Metric
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
*If you like this project or this project help you to explore more about computer vision and machine learning, give it a ⭐ on GitHub!*

This project is still under progress
