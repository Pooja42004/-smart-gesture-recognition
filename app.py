import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import os
import time
import urllib.request
import glob

# Ensure modern layout
st.set_page_config(page_title="Smart Gesture & Sign Recognition", layout="wide")
st.title("🤟 Smart Gesture & Sign Recognition")

# Download the required hand_landmarker.task model automatically if not present
TASK_FILE = "hand_landmarker.task"
if not os.path.exists(TASK_FILE):
    with st.spinner("Downloading MediaPipe hand tracking model... (only happens once)"):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            TASK_FILE
        )

# Initialize MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_FILE),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2)
landmarker = HandLandmarker.create_from_options(options)

# Gesture classes
DATASET_DIR = "dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

def get_classes():
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    return sorted(classes) if classes else []

CLASSES = get_classes()

@st.cache_resource
def load_model():
    if not os.path.exists("gesture_model.h5"):
        return None
    try:
        return tf.keras.models.load_model("gesture_model.h5")
    except Exception as e:
        return None

model = load_model()

def speak(text):
    def run_speech():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            pass
    threading.Thread(target=run_speech, daemon=True).start()

# Drawing function using standard OpenCV
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), 
        (0, 5), (5, 6), (6, 7), (7, 8), 
        (5, 9), (9, 10), (10, 11), (11, 12), 
        (9, 13), (13, 14), (14, 15), (15, 16), 
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for connection in HAND_CONNECTIONS:
                p1 = hand_landmarks[connection[0]]
                p2 = hand_landmarks[connection[1]]
                cv2.line(annotated_image, (int(p1.x * w), int(p1.y * h)), 
                         (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)
            for lm in hand_landmarks:
                cv2.circle(annotated_image, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)
    return annotated_image

def extract_hand_roi(frame, hand_landmarks):
    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    if x_max > x_min and y_max > y_min:
        return frame[y_min:y_max, x_min:x_max]
    return None

tab1, tab2, tab3 = st.tabs(["1. Collect Data", "2. Train Model", "3. Real-Time Recognition"])

with tab1:
    st.header("1. Collect Real-Time Data")
    st.write("Record real-time images using your webcam to build a custom dataset.")
    gesture_name = st.text_input("Enter Gesture Name (e.g., Hello, Yes, No):")
    record_btn = st.button("Start Recording (100 Frames)")
    
    FRAME_WINDOW_COLLECT = st.image([])
    
    if record_btn and gesture_name:
        gesture_dir = os.path.join(DATASET_DIR, gesture_name)
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)
            
        cap = cv2.VideoCapture(0)
        count = 0
        progress_bar = st.progress(0)
        
        while cap.isOpened() and count < 100:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)
            
            drawn_frame = draw_landmarks_on_image(rgb_frame, result)
            FRAME_WINDOW_COLLECT.image(drawn_frame)
            
            if result.hand_landmarks:
                roi = extract_hand_roi(frame, result.hand_landmarks[0])
                if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_resized = cv2.resize(roi_gray, (64, 64))
                    cv2.imwrite(os.path.join(gesture_dir, f"{count}.jpg"), roi_resized)
                    count += 1
                    progress_bar.progress(count / 100)
                    time.sleep(0.02)
        cap.release()
        st.success(f"Successfully collected 100 frames for '{gesture_name}'!")
        time.sleep(1)
        st.rerun()

with tab2:
    st.header("2. Train Model on Custom Dataset")
    st.write("Train the CNN model using the data you just collected.")
    classes = get_classes()
    st.write(f"Available Gestures: **{', '.join(classes) if classes else 'None'}**")
    
    if st.button("Train Model"):
        if len(classes) < 2:
            st.error("Please collect data for at least 2 different gestures in the 'Collect Data' tab.")
        else:
            with st.spinner("Training the model... Please wait."):
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
                
                X, y = [], []
                for idx, cls in enumerate(classes):
                    files = glob.glob(os.path.join(DATASET_DIR, cls, "*.jpg"))
                    for f in files:
                        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = img / 255.0
                            X.append(img)
                            y.append(idx)
                            
                if len(X) == 0:
                    st.error("No images found in the dataset folder.")
                else:
                    X = np.array(X).reshape(-1, 64, 64, 1)
                    y = np.array(y)
                    
                    new_model = Sequential([
                        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
                        MaxPooling2D(2, 2),
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPooling2D(2, 2),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dropout(0.5),
                        Dense(len(classes), activation='softmax')
                    ])
                    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    
                    new_model.fit(X, y, epochs=10, batch_size=16, verbose=0)
                    new_model.save("gesture_model.h5")
                    st.success("Model trained successfully! You can now use Real-Time Recognition.")
                    st.cache_resource.clear()

with tab3:
    st.header("3. Real-Time Recognition")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    recognized_text_placeholder = st.empty()
    
    last_spoken = ""
    last_speech_time = time.time()
    
    if run:
        model = load_model()
        classes = get_classes()
        if model is None or len(classes) == 0:
            st.warning("Model or dataset not found. Please collect data and train the model first.")
        else:
            cap = cv2.VideoCapture(0)
            while cap.isOpened() and run:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect(mp_image)
                
                drawn_frame = draw_landmarks_on_image(rgb_frame, result)
                current_gesture = "None"
                
                if result.hand_landmarks:
                    hand_lms = result.hand_landmarks[0]
                    roi = extract_hand_roi(frame, hand_lms)
                    if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        roi_resized = cv2.resize(roi_gray, (64, 64))
                        roi_normalized = roi_resized / 255.0
                        roi_input = np.reshape(roi_normalized, (1, 64, 64, 1))
                        
                        predictions = model.predict(roi_input, verbose=0)
                        class_id = np.argmax(predictions)
                        confidence = np.max(predictions)
                        
                        if confidence > 0.6 and class_id < len(classes):
                            current_gesture = classes[class_id]
                            
                            h, w, _ = frame.shape
                            x_min = int(min([lm.x for lm in hand_lms]) * w)
                            y_min = int(min([lm.y for lm in hand_lms]) * h)
                            cv2.putText(drawn_frame, f"{current_gesture} ({confidence:.2f})", 
                                        (max(0, x_min), max(0, y_min - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                FRAME_WINDOW.image(drawn_frame)
                
                if current_gesture != "None":
                    recognized_text_placeholder.markdown(f"### 🗣️ Recognized Gesture: **{current_gesture}**")
                    if current_gesture != last_spoken and (time.time() - last_speech_time) > 3.0:
                        speak(current_gesture)
                        last_spoken = current_gesture
                        last_speech_time = time.time()
            cap.release()
