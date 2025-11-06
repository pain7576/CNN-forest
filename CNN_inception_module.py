import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input, \
    concatenate, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd

# Define paths
data_dir = r'D:\MLME project\Mlme\pythonProject\data\Forest2\pca_train_health'  # Update with your actual data path
output_dir = r'D:\MLME project\Mlme\pythonProject\CNN_model_Health_forest1\simple'  # Directory to save models and results

# Parameters
img_width, img_height = 100, 100
batch_size = 50
epochs = 50
learning_rate = 0.0001

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    validation_split=0.2,  # 20% of the data for validation
    rotation_range=20,  # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # Shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # Shift images vertically (fraction of total height)
    shear_range=0.2,  # Shear intensity (shear angle in counter-clockwise direction in degrees)
    zoom_range=0.2,  # Randomly zoom image
    horizontal_flip=True,  # Randomly flip inputs horizontally
    vertical_flip=False,  # Do not flip inputs vertically
    rescale=1. / 255
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)


# Define the Inception module
def inception_module(x, filters):
    branch1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    branch5x5 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(filters[2], (5, 5), padding='same', activation='relu')(branch5x5)

    branch3x3dbl = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    branch3x3dbl = Conv2D(filters[4], (3, 3), padding='same', activation='relu')(branch3x3dbl)
    branch3x3dbl = Conv2D(filters[4], (3, 3), padding='same', activation='relu')(branch3x3dbl)

    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(branch_pool)

    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1)
    return x


# Build the custom CNN model with Inception modules
input_layer = Input(shape=(img_width, img_height, 3))

x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)

x = inception_module(x, [32, 32, 64, 32, 64, 32])
x = MaxPooling2D((2, 2))(x)

x = inception_module(x, [64, 48, 128, 48, 128, 64])
x = MaxPooling2D((2, 2))(x)

x = inception_module(x, [128, 64, 192, 64, 192, 128])
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(4, activation='softmax')(x)  # Adjust based on your number of classes

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and model checkpointing
checkpoint_path = os.path.join(output_dir, 'best_model.h5')

# Ensure the filepath ends with .keras for ModelCheckpoint
if not checkpoint_path.endswith('.keras'):
    checkpoint_path += '.keras'

callbacks = [
    EarlyStopping(patience=15, verbose=1, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Save the final model
final_model_path = os.path.join(output_dir, 'final_model.h5')
model.save(final_model_path)

# Plot training and validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)  # Adjust epochs_range to actual number of epochs

plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title(f'Training and Validation Accuracy\nBatch Size: {batch_size}, Learning Rate: {learning_rate}')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title(f'Training and Validation Loss\nBatch Size: {batch_size}, Learning Rate: {learning_rate}')

plt.tight_layout()
plt.show()

tf.keras.backend.clear_session()
del model
del history
