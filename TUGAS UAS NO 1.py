import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from skimage.feature import hog
from skimage import exposure
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt

# Load your handwritten images
# Assuming you have 10 images (0 to 9) named '0.png', '1.png', ..., '9.png'
handwritten_images = [f"{i}.png" for i in range(10)]

# Extract features using HOG
features, labels = [], []
for image_path in handwritten_images:
    # Load image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    np_image = np.array(image)

    # Extract HOG features
    fd, hog_image = hog(np_image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
    features.append(fd)
    labels.append(int(image_path.split('.')[0]))  # Assuming file names are '0.png', '1.png', ..., '9.png'

    # Plot original image and HOG features
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(np_image, cmap=plt.cm.gray)
    ax[0].set_title('Original Image')

    hog_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax[1].imshow(hog_rescaled, cmap=plt.cm.gray)
    ax[1].set_title('HOG Features')

    plt.show()

features = np.array(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Display predictions
print("Predictions on the test set:")
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
