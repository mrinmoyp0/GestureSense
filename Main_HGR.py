import cv2
import joblib
import mediapipe as mp
import threading
import numpy as np

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize a lock
lock = threading.Lock()

# Load the trained model and scaler
svm_classifier = joblib.load("Model\\gesture_recognition_model.pkl")
scaler = joblib.load("Model\\scaler.pkl")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set frame size to 320x240
cap.set(3, 320)
cap.set(4, 240)

current_gesture = ""  # Variable to store the current recognized gesture

# Function to extract hand landmarks from a frame
def extract_hand_landmarks(frame):
    with lock:
        results = hands.process(frame)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark  # Assuming one hand is present
        return np.array([(landmark.x, landmark.y, landmark.z) for landmark in landmarks]).flatten()
    else:
        return None

# Function to update the gesture recognition result
def update_gesture(frame, scaler, svm_classifier):
    global current_gesture
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Resize the grayscale frame to 320x240 pixels
    resized = cv2.resize(gray, (320, 240))
    
    # Extract hand landmarks as features
    landmarks = extract_hand_landmarks(frame)
    
    if landmarks is not None:
        flattened = landmarks.flatten()
        flattened = scaler.transform([flattened])
        prediction = svm_classifier.predict(flattened)
        
        # Lock to ensure thread safety while updating current_gesture
        with lock:
            current_gesture = prediction[0]

# Function to continuously update gesture recognition result
def gesture_update_thread():
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        
        if not ret:
            break  # Break the loop if frame capture fails
        
        update_gesture(frame, scaler, svm_classifier)

# Start the gesture recognition thread
gesture_thread = threading.Thread(target=gesture_update_thread)
gesture_thread.daemon = True
gesture_thread.start()

while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    
    if not ret:
        break  # Break the loop if frame capture fails
    
    # Display the recognized gesture on the frame
    cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Webcam Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()