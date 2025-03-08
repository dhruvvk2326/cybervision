
# CyberVision: Advanced Visual Malware Classification

Welcome to the CyberVision project! This repository contains the code and resources for the research project titled "CyberVision: Advanced Visual Malware Classification," which explores the use of convolutional neural networks (CNNs) for classifying malware images.

## Overview

CyberVision leverages state-of-the-art CNN architectures, particularly MobileNetV1, to classify malware based on visual representations. The project aims to improve malware detection accuracy and efficiency by utilizing visual features extracted from malware images.

## Dataset

The project uses the Malimg dataset, which consists of 9,339 malware images categorized into 25 different malware families. The dataset is divided as follows:
- Training Images: 7,000
- Validation Images: 1,200
- Test Images: 1,139

## Key Features

- **MobileNetV1 Architecture**: A lightweight and efficient CNN architecture that uses depthwise separable convolutions to reduce computational cost while maintaining high performance.
- **Backpropagation**: Applied to fine-tune the network, enhancing feature extraction and improving classification accuracy.
- **Multiple Classifiers**: Integration with various classifiers including Decision Tree, KNN, SVM, Random Forest, and Naive Bayes to evaluate performance.

## Architecture Diagram

![CNN MobileNetV1 Architecture](https://github.com/abhisheksingh789/CyberVision-Advanced-Visual-Malware-Classification/blob/main/CNN-MobileNetv1.jpg)

## Malimg Dataset Details

![Malimg Dataset](https://github.com/abhisheksingh789/CyberVision-Advanced-Visual-Malware-Classification/blob/main/Malimg_dataset.png)

## Results

### MobileNetV1 Performance
- **Decision Tree**: 
  - Pre-BP Accuracy: 0.9525
  - Post-BP Accuracy: 0.997
- **KNN**: 
  - Pre-BP Accuracy: 0.9954
  - Post-BP Accuracy: 0.9954
- **Naive Bayes**: 
  - Pre-BP Accuracy: 0.9066
  - Post-BP Accuracy: 0.9567
- **Random Forest**: 
  - Pre-BP Accuracy: 0.9964
  - Post-BP Accuracy: 0.9971
- **SVM**: 
  - Pre-BP Accuracy: 1.0016
  - Post-BP Accuracy: 1.0016

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Matplotlib


