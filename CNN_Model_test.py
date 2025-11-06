import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Define paths
data_dir = r'D:\MLME project\Mlme\pythonProject\data\Forest2\cnn_train_health'  # Update with your actual data path
model_path = r'D:\MLME project\Mlme\pythonProject\CNN_model_Health_forest2\CNN Models\best model after augmentation\best_model.h5.keras'  # Path to your saved model
test_data_dir = r'D:\MLME project\Mlme\pythonProject\data\Forest2\cnn_test_health'  # Update with your actual test data path

# Parameters
img_width, img_height = 100, 100
batch_size = 10

# Data augmentation and preprocessing for test data
test_datagen = ImageDataGenerator(
            rescale=1. / 255
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Print the class indices to debug
print(f"Class indices from test data: {test_generator.class_indices}")

# Manually set the class indices to match those expected by the model
expected_class_indices = {'H': 0, 'HD': 1, 'LD': 2, 'other': 3}
test_generator.class_indices = expected_class_indices

# Ensure the generator uses the correct classes
test_generator.classes = np.array([expected_class_indices[os.path.dirname(filename)] for filename in test_generator.filenames])

# Load the saved model
model = load_model(model_path)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Plotting the evaluation results
labels = list(expected_class_indices.keys())
predictions = model.predict(test_generator, steps=np.ceil(test_generator.samples / batch_size))
predicted_classes = predictions.argmax(axis=1)
true_classes = test_generator.classes

# Confusion matrix and classification report (optional)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print('Classification Report')
print(classification_report(true_classes, predicted_classes, target_names=labels))
