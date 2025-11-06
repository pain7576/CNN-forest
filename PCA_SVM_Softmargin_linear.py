import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
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
r = 2 # Or whatever number you choose

# Extract the first r principal components
e_vec_subset = sorted_e_vec[:, :r]

# Compress test data
X_test_compressed = X_test @ e_vec_subset

# Print compression result
print(f'You have reduced the memory footprint by {100 - (r / 64 * 100):2.2f}% while retaining {info[r-1] * 100:2.2f}% of the information!')

# Compress training data
X_train_compressed = X_train @ e_vec_subset

# Create and train the Linear SVM classifier with soft margin
# C parameter controls the trade-off between margin and misclassification
linear_svc = LinearSVC(C=1.0, random_state=42)
linear_svc.fit(X_train_compressed, y_train_encoded)

# For visualization, use only the first 2 components
X_train_compressed_2d = X_train_compressed[:, :2]
X_test_compressed_2d = X_test_compressed[:, :2]

# Create a mesh to plot in
x_min, x_max = X_train_compressed_2d[:, 0].min() - 1, X_train_compressed_2d[:, 0].max() + 1
y_min, y_max = X_train_compressed_2d[:, 1].min() - 1, X_train_compressed_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Initialize plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the decision boundary
Z = linear_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

# Create a color bar
cbar = plt.colorbar(cs)
cbar.set_label('Predicted Class')
cbar.set_ticks(np.unique(Z))
cbar.set_ticklabels(le.inverse_transform(np.unique(Z)))

# Plot the training points
for i, class_label in enumerate(np.unique(y_train_encoded)):
    ax.scatter(X_train_compressed_2d[y_train_encoded == class_label, 0],
               X_train_compressed_2d[y_train_encoded == class_label, 1],
               c=plt.cm.RdYlBu(i / (len(np.unique(y_train_encoded)) - 1)),
               marker='x', edgecolors='black',
               label=f'Train {le.inverse_transform([class_label])[0]}')

# Plot the testing points
for i, class_label in enumerate(np.unique(y_test_encoded)):
    ax.scatter(X_test_compressed_2d[y_test_encoded == class_label, 0],
               X_test_compressed_2d[y_test_encoded == class_label, 1],
               c=plt.cm.RdYlBu(i / (len(np.unique(y_test_encoded)) - 1)),
               marker='s', edgecolors='black',
               label=f'Test {le.inverse_transform([class_label])[0]}')

# Plot the decision boundary
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Post-processing
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_title('Linear Soft Margin SVM Classification (First 2 PCA Components)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Display the plot
plt.tight_layout()
plt.show()

# Print classification report
y_pred = linear_svc.predict(X_test_compressed)
print(classification_report(y_test_encoded, y_pred))

# Print accuracy score
print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred):.4f}")