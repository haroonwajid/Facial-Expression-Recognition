# Facial Expression Recognition

This project leverages **deep learning** to automatically recognize facial expressions and predict continuous affective states (valence and arousal). It combines a **custom CNN** with multiple **pretrained models** to provide both categorical and continuous emotion recognition.

---

## Features

* Custom CNN implementation for emotion recognition.
* Pretrained CNN baselines: **ResNet18, VGG16, EfficientNetB0, MobileNetV2, DenseNet121**.
* Classification of **8 emotion categories**: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt.
* Prediction of **valence** (pleasant–unpleasant) and **arousal** (calm–excited) in continuous domain.
* Comprehensive evaluation using categorical and continuous metrics.
* Visualization of training progress and sample predictions.

---

## Dataset

The dataset includes:

* Cropped and resized facial images (**224×224 RGB**).
* **68 facial landmarks** for each image.
* **Emotion labels** (0–7).
* **Valence and arousal values** in the range [-1, +1].

Note: Dataset files are stored in `.npy` format (images, labels, annotations).

---

## Model Architectures

* **Custom CNN**: Built from scratch with convolutional, pooling, and dense layers.
* **Pretrained Models** (with fine-tuned final layers):

  * ResNet18
  * VGG16
  * EfficientNetB0
  * MobileNetV2
  * DenseNet121

---

## Training Setup

* **Framework**: PyTorch
* **Batch Size**: 32
* **Optimizer**: Adam / SGD
* **Loss Functions**:

  * CrossEntropyLoss (categorical)
  * MSELoss (continuous)
* **Learning Rate**: 1e-4 (tuned per model)
* **Epochs**: 30–50 (depending on convergence)

---

## Evaluation Metrics

### Categorical

* Accuracy
* F1-Score
* Cohen’s Kappa
* Krippendorff’s Alpha
* AUC / AUC-PR

### Continuous

* RMSE (Root Mean Square Error)
* CORR (Pearson Correlation)
* SAGR (Sign Agreement Metric)
* CCC (Concordance Correlation Coefficient)

---

## Results

* Training and validation curves show decreasing loss and increasing accuracy.
* Pretrained models outperform the custom CNN, with **DenseNet121 and EfficientNetB0** achieving the best overall results.
* Continuous domain metrics indicate strong correlation and concordance, suitable for real-world deployment.

---

## Sample Predictions

The notebook includes visualizations of:

* Correctly classified samples.
* Misclassified samples with predicted vs. actual labels.

---

## Future Work

* Expand dataset with more diverse expressions.
* Apply data augmentation for robustness.
* Explore transformer-based vision models (ViT, Swin).
* Real-time emotion recognition pipeline.

