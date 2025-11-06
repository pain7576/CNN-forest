import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd

# Define paths
data_dir = r'D:\MLME project\Mlme\pythonProject\data\Forest2\pca_train_health'
output_dir = r'D:\MLME project\Mlme\pythonProject\CNN_model_Health_forest1\simple'

# Parameters
img_width, img_height = 100, 100
epochs = 50

# Arrays of batch sizes and learning rates
batch_sizes =[40,50]
learning_rates = [0.1, 0.01]


# Function to create data generators
def create_data_generators(batch_size):
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
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

    return train_generator, validation_generator


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
def create_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

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
    output_layer = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output_layer)


# Results storage
results = []

# Main training loop
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        print(f"Training with batch size: {batch_size}, learning rate: {learning_rate}")

        # Create data generators
        train_generator, validation_generator = create_data_generators(batch_size)

        # Create and compile the model
        model = create_model((img_width, img_height, 3), num_classes=4)
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Callbacks
        checkpoint_path = os.path.join(output_dir, f'best_model_bs{batch_size}_lr{learning_rate}.keras')
        callbacks = [
            EarlyStopping(patience=1, verbose=1, restore_best_weights=True),
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

        # Store results
        results.append({
            'Batch Size': batch_size,
            'Learning Rate': learning_rate,
            'Final Training Accuracy': history.history['accuracy'][-1],
            'Final Validation Accuracy': history.history['val_accuracy'][-1],
            'Final Training Loss': history.history['loss'][-1],
            'Final Validation Loss': history.history['val_loss'][-1],
            'Epochs Completed': len(history.history['accuracy'])
        })

        # Plot training and validation accuracy/loss
        plt.figure(figsize=(12, 8))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title(f'Training and Validation Accuracy\nBatch Size: {batch_size}, Learning Rate: {learning_rate}')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title(f'Training and Validation Loss\nBatch Size: {batch_size}, Learning Rate: {learning_rate}')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_bs{batch_size}_lr{learning_rate}.png'))
        plt.close()

        # Clear session to free up memory
        tf.keras.backend.clear_session()
        del model
        del history

# Save results to Excel
results_df = pd.DataFrame(results)
results_df.to_excel(os.path.join(output_dir, 'training_results.xlsx'), index=False)

print("Training completed. Results saved to 'training_results.xlsx'.")