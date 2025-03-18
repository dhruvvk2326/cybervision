# %%
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import keras.utils as image
from tensorflow.keras.applications import MobileNet

from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from sklearn.metrics import matthews_corrcoef
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping



# %%
# Define directories
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
train_dir = 'C:/Users/dhruv/Desktop/project/CyberVision-Advanced-Visual-Malware-Classification-main/malimg_dataset/train'
test_dir = 'C:/Users/dhruv/Desktop/project/CyberVision-Advanced-Visual-Malware-Classification-main/malimg_dataset/test'
val_dir = 'C:/Users/dhruv/Desktop/project/CyberVision-Advanced-Visual-Malware-Classification-main/malimg_dataset/val'

# %%
image_size = (224, 224, 3)
batch_size=32

# %%
def load_images(directory):
    images = []
    labels = []
    for label, class_name in enumerate(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):  # Check if it's a directory
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                img = image.load_img(img_path, target_size=image_size)
                img_array = image.img_to_array(img)
                images.append(img_array)
                labels.append(label)
    return np.array(images), np.array(labels)

# %%
# Load training and test images
train_images, train_labels = load_images(train_dir)
test_images, test_labels = load_images(test_dir)
val_images, val_labels = load_images(val_dir)


# %%
print(train_labels)
# print((train_images[0][0]))

# %%
# Preprocess images
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
val_images = val_images.astype('float32') / 255.0

# %%
# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
train_datagen.fit(train_images)


# %%
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=image_size)
for layer in base_model.layers[:-5]:  # Fine-tune top 5 layers
    layer.trainable = False

# %%
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer='l2')(x)
x = Dropout(0.5)(x)
predictions = Dense(25, activation='softmax')(x)

# %%
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with AdamW optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer=AdamW(learning_rate=1e-4), metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# %%
history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size),
    epochs=50,  # Adjust as needed
    validation_data=(val_images, val_labels),  # If using a validation set
    callbacks=[early_stopping]
)


# %%
model.summary()

# %%
# Train the model (adjust epochs and batch size as needed)
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# %%
# Extract features using MobileNetV1
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
# tf.keras.backend.clear_session()

# %%
# Extract features using the model without the top layer in batches
# Get the penultimate layer's output
feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)


batch_size=32
# Use the extract_features function to process images in batches
train_features = extract_features(feature_extractor, train_images, batch_size)
test_features = extract_features(feature_extractor, test_images, batch_size)

print("Shape of extracted train features:", train_features.shape)
print("Shape of extracted test features:", test_features.shape)

# %%
# Extract features using the model without the top layer
# train_features = mobilenet_with_gap.predict(train_images)
# test_features = mobilenet_with_gap.predict(test_images)

# %%
print(train_features.shape)
print(test_features.shape)

# %%
# Define the parameter grid for Decision Tree
# dt_param_grid = {'criterion': ['entropy'], 'max_depth': [ 10]}
# # n_estimators:500, max_depth:10, criterion:'entropy'
# Define the expanded parameter grid for Decision Tree
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],  # Add more depth options
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Use GridSearchCV to find the best parameters for Decision Tree
dt_grid_search = GridSearchCV(dt_classifier, dt_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
dt_grid_search.fit(train_features, train_labels)

# Get the best parameters for Decision Tree
best_dt_params = dt_grid_search.best_params_

# Use the best parameters to train the final Decision Tree model
final_dt_classifier = DecisionTreeClassifier(**best_dt_params)
start_time_dt = time.time()
final_dt_classifier.fit(train_features, train_labels)
train_time_dt = time.time() - start_time_dt

# Predict labels for the test set using Decision Tree
dt_predictions = final_dt_classifier.predict(test_features)

# Evaluate accuracy for Decision Tree
dt_accuracy = accuracy_score(test_labels, dt_predictions)
print("\nBest Decision Tree Parameters:", best_dt_params)
print("Decision Tree Accuracy:", dt_accuracy)


# %%
# Additional evaluation metrics for Decision Tree
precision_dt = precision_score(test_labels, dt_predictions, average='weighted')
recall_dt = recall_score(test_labels, dt_predictions, average='weighted')
f1_dt = f1_score(test_labels, dt_predictions, average='weighted')

print("Train Time (sec) Decision Tree:", train_time_dt)
# If you used GridSearchCV, you can access the training time with `cv_results_`
grid_train_time_dt = dt_grid_search.cv_results_['mean_fit_time'][dt_grid_search.best_index_]
print("Train Time (sec) Grid Decision Tree:", grid_train_time_dt)

# Calculate Matthews Correlation Coefficient (MCC) for Decision Tree
mcc_dt = matthews_corrcoef(test_labels, dt_predictions)
print("Matthews Correlation Coefficient (MCC) Decision Tree:", mcc_dt)

# Accuracy for optimization algorithm (GridSearchCV)
optimization_accuracy = dt_grid_search.best_score_
print("Optimization Algorithm Accuracy:", optimization_accuracy)

# %%
# Confusion matrix for Decision Tree
conf_matrix_dt = confusion_matrix(test_labels, dt_predictions)
precision_dt = precision_score(test_labels, dt_predictions, average='weighted')

FP_dt = conf_matrix_dt.sum(axis=0) - np.diag(conf_matrix_dt)  
FN_dt = conf_matrix_dt.sum(axis=1) - np.diag(conf_matrix_dt)
TP_dt = np.diag(conf_matrix_dt)
TN_dt = conf_matrix_dt.sum()


# %%
# Display results for Decision Tree
print("\nResults for Decision Tree:")
print("Precision (Decision Tree):", precision_dt)
print("Recall (Decision Tree):", recall_dt)
print("F1 Score (Decision Tree):", f1_dt)
print("True Positive Rate (TPR) (Decision Tree):", np.mean(TP_dt / (TP_dt + FN_dt)))
print("True Negative Rate (TNR) (Decision Tree):", np.mean(TN_dt / (TN_dt + FP_dt)))
print("False Positive Rate (FPR) (Decision Tree):", np.mean(FP_dt / (FP_dt + TN_dt)))
print("False Negative Rate (FNR) (Decision Tree):", np.mean(FN_dt / (TP_dt + FN_dt)))

print("False Discovery Rate (FDR) (Decision Tree):", np.mean(FP_dt / (FP_dt + TP_dt)))
print("False Omission Rate (FOR) (Decision Tree):", np.mean(FN_dt / (FN_dt + TN_dt)))

print("Matthews Correlation Coefficient (MCC) (Decision Tree):", mcc_dt)

# %%
conf_matrix_dt


# %%
# Plot TPR, TNR, FPR, FNR for Decision Tree
fig, ax = plt.subplots(figsize=(15, 10))

bar_width = 0.2
index = np.arange(25)  # Assuming you have 25 classes, adjust this based on your data



bar1 = ax.bar(index, TP_dt, bar_width, label='TPR')
bar2 = ax.bar(index + bar_width, TN_dt, bar_width, label='TNR')
bar3 = ax.bar(index + 2 * bar_width, FP_dt, bar_width, label='FPR')
bar4 = ax.bar(index + 3 * bar_width, FN_dt, bar_width, label='FNR')

ax.set_xlabel('Class')
ax.set_ylabel('Scores')
ax.set_title('Comparison of TPR, TNR, FPR, FNR for Decision Tree')
ax.set_xticks(index + 1.5 * bar_width)
ax.legend()

plt.show()

# %%
# Plot Confusion Matrix for Decision Tree
fig, ax = plt.subplots(figsize=(20, 20))
sns.set(font_scale=1.2)  # Adjust font size for better readability
disp = ConfusionMatrixDisplay(conf_matrix_dt, display_labels=np.unique(test_labels))
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix for Decision Tree')
plt.show()


