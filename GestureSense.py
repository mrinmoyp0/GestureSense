import streamlit as st
from streamlit_lottie import st_lottie  # Import Streamlit Lottie
import json

# Load the Lottie animation from a JSON file
with open("animation.json", "r") as f:
    lottie_json = json.load(f)

# Function to display Lottie animation
def display_animation():
    st_lottie(lottie_json, speed=1, width=700, height=150)

# Display the Lottie animation immediately after running the app
display_animation()

#st.title("Hand Gesture Recognition")
# Center-align the title using Markdown and HTML
st.markdown("<h1 style='text-align: center;'>GestureSense👌</h1>", unsafe_allow_html=True)

import cv2
import joblib
import mediapipe as mp
import numpy as np
import threading
from PIL import Image

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize a lock
lock = threading.Lock()

# Load the trained model and scaler
svm_classifier = joblib.load("Model\\gesture_recognition_model.pkl")
scaler = joblib.load("Model\\scaler.pkl")

# Initialize webcam and thread-related variables
cap = None
current_gesture = ""
webcam_started = False  # Flag to track if the webcam is started

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
def update_gesture(frame):
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
    global cap
    global webcam_started
    while webcam_started:
        ret, frame = cap.read()  # Capture a frame from the webcam

        if not ret:
            break  # Break the loop if frame capture fails

        update_gesture(frame)

# Streamlit UI
# Create a dropdown menu for choosing the recognition mode
recognition_mode = st.selectbox("Select Recognition Mode", ("Live Hand Gesture Recognition", "Hand Recognition from Image"))


# Initialize the frame placeholder
frame_placeholder = st.empty()

if recognition_mode == "Live Hand Gesture Recognition":
    # Create Start and Stop buttons for live recognition
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    # Start the gesture recognition thread when the Start button is clicked
    if start_button and not webcam_started:
        cap = cv2.VideoCapture(0)
        cap.set(3, 320)
        cap.set(4, 240)

        webcam_started = True

        gesture_thread = threading.Thread(target=gesture_update_thread)
        gesture_thread.daemon = True
        gesture_thread.start()

    # Display the webcam feed and recognized gesture when the webcam is running
    if cap is not None and webcam_started:
        while webcam_started:
            ret, frame = cap.read()  # Capture a frame from the webcam

            if not ret:
                break  # Break the loop if frame capture fails

            # Display the recognized gesture on the frame
            cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame using Streamlit
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    # Stop the webcam and release resources when the Stop button is clicked
    if stop_button:
        if cap is not None and cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
        webcam_started = False  # Prevent the code from restarting automatically

elif recognition_mode == "Hand Recognition from Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Process the uploaded image for hand gesture recognition
        image = Image.open(uploaded_image)
        image = np.array(image)

        # Perform hand gesture recognition on the image
        update_gesture(image)

        # Display the recognized gesture
        st.image(image, channels="BGR", use_column_width=True)
        st.write(f"Recognized Gesture: {current_gesture}")