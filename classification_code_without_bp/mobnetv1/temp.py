import os
import numpy as np
import keras.utils as image
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam  # Import from TensorFlow directly
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator

# Load MobileNet with some layers set as trainable
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Add new layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(25, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with legacy optimizer
model.compile(optimizer=Adam(learning_rate=0.0001, name='Adam'), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    'C:/Users/dhruv/Desktop/project/CyberVision-Advanced-Visual-Malware-Classification-main/malimg_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(
    'C:/Users/dhruv/Desktop/project/CyberVision-Advanced-Visual-Malware-Classification-main/malimg_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Extract features
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)
features_train = feature_extractor.predict(train_generator)
features_val = feature_extractor.predict(validation_generator)

# Train Decision Tree
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=2, min_samples_split=5)
tree_model.fit(features_train, train_generator.classes)

# Evaluate Decision Tree
predictions = tree_model.predict(features_val)
accuracy = accuracy_score(validation_generator.classes, predictions)
print('Decision Tree Accuracy:', accuracy)
