import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import random
import mediapipe as mp
import cv2

# Initialize empty lists for data and labels
data = []
labels = []

# Define the path to your dataset directory
dataset_dir = "Image_Data"

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to extract hand landmarks from a frame
def extract_hand_landmarks(frame):
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark  # Assuming one hand is present
        return np.array([(landmark.x, landmark.y, landmark.z) for landmark in landmarks]).flatten()
    else:
        return None

# Iterate through each subdirectory (class) in the dataset directory
for sub_dir in os.listdir(dataset_dir):
    class_label = sub_dir
    sub_dir_path = os.path.join(dataset_dir, sub_dir)
    
    if os.path.isdir(sub_dir_path):  # Check if it's a directory
        # List all image files in the subdirectory
        image_files = os.listdir(sub_dir_path)
        
        # Randomly select 30 images from each class
        num_images_to_select = min(120, len(image_files))
        selected_images = random.sample(image_files, num_images_to_select)

        # Iterate through the selected images
        for image_file in selected_images:
            image_path = os.path.join(sub_dir_path, image_file)
            image = cv2.imread(image_path)  # Read the image
            
            # Resize the image to 320x240 pixels
            image = cv2.resize(image, (320, 240))
            
            # Extract hand landmarks as features
            landmarks = extract_hand_landmarks(image)
            
            if landmarks is not None:
                data.append(landmarks)
                labels.append(class_label)

# Convert lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVM classifier
svm_classifier = svm.SVC(kernel='linear')

# Train the SVM model on the training data
svm_classifier.fit(X_train, y_train)

# Save the trained model to a custom file path
model_save_path = "Model\\gesture_recognition_model.pkl"
joblib.dump(svm_classifier, model_save_path)

# Save the scaler to a custom file path
scaler_save_path = "Model\\scaler.pkl"
joblib.dump(scaler, scaler_save_path)