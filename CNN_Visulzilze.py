import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Define the path to your saved model
model_path = r'D:\MLME project\Mlme\pythonProject\CNN_model_Health_forest1\simple\best model befor augmentation\best_model.h5.keras'

# Load the model
model = load_model(model_path)

# Define the path to save the model visualization
visualization_path = r'D:\MLME project\Mlme\pythonProject\CNN_model_Health_forest1\simple\model_visualization.png'

# Plot the model
plot_model(model, to_file=visualization_path, show_shapes=True, show_layer_names=True)

print(f"Model visualization saved to {visualization_path}")
