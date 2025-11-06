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
r = 10

# Extract the first r principal components
e_vec_subset = sorted_e_vec[:, :r]

# Compress test data
X_test_compressed = X_test @ e_vec_subset

# Print compression result
print(f'You have reduced the memory footprint by {100 - (r / 64 * 100):2.2f}% while retaining {info[r-1] * 100:2.2f}% of the information!')

# Compress training data
X_train_compressed = X_train @ e_vec_subset

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_train_compressed)

# Predict clusters for training and test data
train_clusters = kmeans.predict(X_train_compressed)
test_clusters = kmeans.predict(X_test_compressed)

# Calculate accuracy for training and test data
train_accuracy = accuracy_score(y_train_encoded, train_clusters)
test_accuracy = accuracy_score(y_test_encoded, test_clusters)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


def visualize_kmeans_clusters(X_compressed, clusters, true_labels, title):
    """
    Visualize the K-means clustering results.

    Args:
    X_compressed: Compressed data (n_samples, n_components)
    clusters: Predicted cluster labels
    true_labels: True class labels
    title: Title for the plot
    """
    plt.figure(figsize=(12, 5))

    # Plot K-means clusters
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_compressed[:, 0], X_compressed[:, 1], c=clusters, cmap='viridis')
    plt.title(f'K-means Clustering: {title}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster')

    # Plot true labels
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_compressed[:, 0], X_compressed[:, 1], c=true_labels, cmap='viridis')
    plt.title(f'True Labels: {title}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='True Class')

    plt.tight_layout()
    plt.show()


# After the existing code, add these lines to visualize the K-means clustering results
print("\nVisualizing K-means clustering results for training data:")
visualize_kmeans_clusters(X_train_compressed, train_clusters, y_train_encoded, "Training Data")

print("\nVisualizing K-means clustering results for test data:")
visualize_kmeans_clusters(X_test_compressed, test_clusters, y_test_encoded, "Test Data")
