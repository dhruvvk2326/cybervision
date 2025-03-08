# %%
# %%
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import keras.utils as image
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from sklearn.metrics import matthews_corrcoef
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# %%
# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%
# Define directories
train_dir = 'C:/Users/dhruv/Desktop/project/CyberVision-Advanced-Visual-Malware-Classification-main/malimg_dataset/train'
test_dir = 'C:/Users/dhruv/Desktop/project/CyberVision-Advanced-Visual-Malware-Classification-main/malimg_dataset/test'


# %%
# %%
image_size = (224, 224, 3)

# %%
# Function to load images
def load_images(directory):
    images = []
    labels = []
    for label, class_name in enumerate(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                img = image.load_img(img_path, target_size=image_size)
                img_array = image.img_to_array(img)
                images.append(img_array)
                labels.append(label)
    return np.array(images), np.array(labels)

# %%
# %%
# Load training and test images
train_images, train_labels = load_images(train_dir)
test_images, test_labels = load_images(test_dir)

# Preprocess images
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# %%
# Load ResNet50 with pre-trained weights on ImageNet
base_resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='None')

# Add Global Average Pooling (GAP) layer
x = base_resnet50.output
x = GlobalAveragePooling2D()(x)

# Create the model
resnet50_with_gap = Model(inputs=base_resnet50.input, outputs=x)

# %%
# Function to extract features
def extract_features(model, images, batch_size):
    num_images = images.shape[0]
    features = []
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch = images[start_idx:end_idx]
        batch_features = model.predict(batch)
        features.append(batch_features)
    return np.concatenate(features)


# %%
batch_size = 8
# Extract features using ResNet50
train_features = extract_features(resnet50_with_gap, train_images, batch_size)
test_features = extract_features(resnet50_with_gap, test_images, batch_size)


# %%
# %%
# Define the parameter grid for SVM
svm_param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

# Create SVM classifier
svm_classifier = SVC()

# Use GridSearchCV to find the best parameters for SVM
svm_grid_search = GridSearchCV(svm_classifier, svm_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
svm_grid_search.fit(train_features, train_labels)

# Get the best parameters for SVM
best_svm_params = svm_grid_search.best_params_

# Train the final SVM model with the best parameters
final_svm_classifier = SVC(**best_svm_params)
start_time_svm = time.time()
final_svm_classifier.fit(train_features, train_labels)
train_time_svm = time.time() - start_time_svm

# Predict labels for the test set using SVM
svm_predictions = final_svm_classifier.predict(test_features)

# Evaluate accuracy for SVM
svm_accuracy = accuracy_score(test_labels, svm_predictions)
print("\nBest SVM Parameters:", best_svm_params)
print("SVM Accuracy:", svm_accuracy)

# %%
# %%
# Additional evaluation metrics for SVM
precision_svm = precision_score(test_labels, svm_predictions, average='weighted')
recall_svm = recall_score(test_labels, svm_predictions, average='weighted')
f1_svm = f1_score(test_labels, svm_predictions, average='weighted')

print("Train Time (sec) SVM:", train_time_svm)
grid_train_time_svm = svm_grid_search.cv_results_['mean_fit_time'][svm_grid_search.best_index_]
print("Train Time (sec) Grid SVM:", grid_train_time_svm)

# Calculate Matthews Correlation Coefficient (MCC) for SVM
mcc_svm = matthews_corrcoef(test_labels, svm_predictions)
print("Matthews Correlation Coefficient (MCC) SVM:", mcc_svm)

# Accuracy for optimization algorithm (GridSearchCV)
optimization_accuracy_svm = svm_grid_search.best_score_
print("Optimization Algorithm Accuracy SVM:", optimization_accuracy_svm)

# %%
# %%
# Confusion matrix for SVM
conf_matrix_svm = confusion_matrix(test_labels, svm_predictions)
precision_svm = precision_score(test_labels, svm_predictions, average='weighted')

FP_svm = conf_matrix_svm.sum(axis=0) - np.diag(conf_matrix_svm)
FN_svm = conf_matrix_svm.sum(axis=1) - np.diag(conf_matrix_svm)
TP_svm = np.diag(conf_matrix_svm)
TN_svm = conf_matrix_svm.sum()

# %%
# Display results for SVM
print("\nResults for SVM:")
print("Precision (SVM):", precision_svm)
print("Recall (SVM):", recall_svm)
print("F1 Score (SVM):", f1_svm)
print("True Positive Rate (TPR) (SVM):", np.mean(TP_svm / (TP_svm + FN_svm)))
print("True Negative Rate (TNR) (SVM):", np.mean(TN_svm / (TN_svm + FP_svm)))
print("False Positive Rate (FPR) (SVM):", np.mean(FP_svm / (FP_svm + TN_svm)))
print("False Negative Rate (FNR) (SVM):", np.mean(FN_svm / (TP_svm + FN_svm)))
print("False Discovery Rate (FDR) (SVM):", np.mean(FP_svm / (FP_svm + TP_svm)))
print("False Omission Rate (FOR) (SVM):", np.mean(FN_svm / (FN_svm + TN_svm)))
print("Matthews Correlation Coefficient (MCC) (SVM):", mcc_svm)

# %%
# Plot TPR, TNR, FPR, FNR
fig, ax = plt.subplots(figsize=(15, 10))

bar_width = 0.2
index = np.arange(25)  # Assuming you have 25 classes, adjust this based on your data

bar1 = ax.bar(index, TP_svm, bar_width, label='TPR')
bar2 = ax.bar(index + bar_width, TN_svm, bar_width, label='TNR')
bar3 = ax.bar(index + 2 * bar_width, FP_svm, bar_width, label='FPR')
bar4 = ax.bar(index + 3 * bar_width, FN_svm, bar_width, label='FNR')

ax.set_xlabel('Class')
ax.set_ylabel('Scores')
ax.set_title('Comparison of TPR, TNR, FPR, FNR for knn')
ax.set_xticks(index + 1.5 * bar_width)
ax.legend()

plt.show()

# %%
# %%
# Plot Confusion Matrix for SVM
fig, ax = plt.subplots(figsize=(20, 20))
sns.set(font_scale=1.2)  # Adjust font size for better readability
disp = ConfusionMatrixDisplay(conf_matrix_svm, display_labels=np.unique(test_labels))
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix for SVM')
plt.show()



