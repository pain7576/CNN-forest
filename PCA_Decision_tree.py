import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
# Define data folder paths
data_folder = r'D:\MLME project\Mlme\pythonProject\data'
forest2_train_folder = os.path.join(data_folder, 'Forest1', 'pca_train_health')
forest2_test_folder = os.path.join(data_folder, 'Forest1', 'pca_test_health')

# Function to load images and labels from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for class_name in ['H', 'HD', 'LD', 'other']:
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        labels.append(class_name)
    return np.array(images), np.array(labels)

# Load images and labels for training and testing
X_train_forest2, y_train_forest2 = load_images_from_folder(forest2_train_folder)
X_test_forest2, y_test_forest2 = load_images_from_folder(forest2_test_folder)

# Convert labels to numerical values
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_forest2)
y_test_encoded = le.transform(y_test_forest2)

# Compute mean and standard deviation of raw data
X_raw_mean_train = np.mean(X_train_forest2, axis=0)
X_raw_mean_test = np.mean(X_test_forest2, axis=0)
X_raw_std_train = np.std(X_train_forest2, axis=0)
X_raw_std_test = np.std(X_test_forest2, axis=0)

# Get the index of the pixels/columns where the standard deviation is not zero
idx_train = X_raw_std_train != 0
idx_test = X_raw_std_test != 0

# Delete columns with zero information/standard deviation/variance
X_red_train = X_train_forest2[:, idx_train]
X_red_test = X_test_forest2[:, idx_test]

# Compute mean and standard deviation of the cleaned data
X_red_mean_train = np.mean(X_red_train, axis=0)
X_red_std_train = np.std(X_red_train, axis=0)

# Standardize data (avoid division by zero)
epsilon = 1e-8
X_train = (X_red_train - X_red_mean_train) / (X_red_std_train + epsilon)

X_red_mean_test = np.mean(X_red_test, axis=0)
X_red_std_test = np.std(X_red_test, axis=0)

# Standardize test data (avoid division by zero)
X_test = (X_red_test - X_red_mean_test) / (X_red_std_test + epsilon)

# Compute covariance matrix
cov_mat = np.cov(X_train.T)

# Compute eigenvalues and eigenvectors
e_val, e_vec = np.linalg.eig(cov_mat)

# Ensure values are real
e_val = np.real(e_val)
e_vec = np.real(e_vec)

# Get sorting indices from largest to smallest eigenvalue
sorted_index = np.argsort(e_val)[::-1]

# Sort the eigenvalues and eigenvectors according to the index
sorted_e_val = e_val[sorted_index]
sorted_e_vec = e_vec[:, sorted_index]

# Total information
total_info = np.sum(sorted_e_val)

# Information for all values of r
info = [np.sum(sorted_e_val[:i + 1]) / total_info for i in range(sorted_e_val.shape[0])]

# Plot the information
plt.figure()
plt.plot(info, 'x')
plt.xlabel('$r$')
plt.ylabel('$\iota$')
plt.title('Information Retained vs Number of Components')
plt.tight_layout()
plt.show()

# Number of first principal components used for compression
r = 2

# Extract the first r principal components
e_vec_subset = sorted_e_vec[:, :r]

# Compress test data
X_test_compressed = X_test @ e_vec_subset

# Print compression result
print(f'You have reduced the memory footprint by {100 - (r / 64 * 100):2.2f}% while retaining {info[r-1] * 100:2.2f}% of the information!')

# Compress training data
X_train_compressed = X_train @ e_vec_subset

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Train Decision Tree classifier
dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
dt_classifier.fit(X_train_compressed, y_train_encoded)

# Predict on training and test data
dt_train_pred = dt_classifier.predict(X_train_compressed)
dt_test_pred = dt_classifier.predict(X_test_compressed)

# Calculate accuracies
dt_train_accuracy = accuracy_score(y_train_encoded, dt_train_pred)
dt_test_accuracy = accuracy_score(y_test_encoded, dt_test_pred)

print("\nDecision Tree Results:")
print(f"Training Accuracy: {dt_train_accuracy:.4f}")
print(f"Test Accuracy: {dt_test_accuracy:.4f}")


def plot_decision_boundary_dt_low_res(X, y, classifier, title, resolution=0.2):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Generate a grid of points with lower resolution
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # Predict the function value for the whole grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    # Create a legend
    legend_elements = [plt.scatter([], [], c='r', label='H'),
                       plt.scatter([], [], c='b', label='HD'),
                       plt.scatter([], [], c='g', label='LD'),
                       plt.scatter([], [], c='y', label='other')]
    plt.legend(handles=legend_elements, loc='best')
    plt.title(title)
    plt.show()

# Use the low-resolution function for visualization
print("\nVisualizing Decision Tree decision boundary for training data (low resolution):")
plot_decision_boundary_dt_low_res(X_train_compressed, y_train_encoded, dt_classifier, "Decision Tree Boundary - Training Data", resolution=0.2)

print("\nVisualizing Decision Tree decision boundary for test data (low resolution):")
plot_decision_boundary_dt_low_res(X_test_compressed, y_test_encoded, dt_classifier, "Decision Tree Boundary - Test Data", resolution=0.2)