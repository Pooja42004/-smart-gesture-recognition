# Smart Gesture & Sign Recognition Assistant

## Problem Statement

Many people with hearing and speech disabilities use sign language for communication. However, most people do not understand sign language, creating communication barriers. 

Our project solves this problem by automatically recognizing hand gestures and converting them into understandable text and voice output.

## Main Idea

- The webcam captures hand gestures.
- **MediaPipe** detects the hand landmarks, and the **CNN model** predicts the gesture.
- The recognized gesture is then displayed as text and converted into speech.

## Technologies Used

- **Python** for programming
- **OpenCV** for webcam handling
- **MediaPipe** for hand tracking
- **CNN using TensorFlow/Keras** for gesture recognition
- **pyttsx3** for text-to-speech conversion
- **Streamlit** for user interface

## Setup Instructions

1. **Install Python**
   Ensure you have Python 3.8 to 3.10 installed on your system (TensorFlow is best supported on these versions).

2. **Open Command Prompt / Terminal**
   Navigate to the project folder on your D drive:
   ```cmd
   D:
   cd D:\SmartGestureRecognition
   ```

3. **Install Dependencies**
   Run the following command to install the required libraries:
   ```cmd
   pip install -r requirements.txt
   ```

4. **Generate the Gesture Model**
   Because training a CNN requires a large dataset of images, we have provided a script that creates a structural "dummy" CNN model. Run the following command to generate the `gesture_model.h5` file:
   ```cmd
   python gesture_model.py
   ```
   *(Note: For real-world use, you would train this model on a labeled dataset of gesture images. The current model is an untrained placeholder to demonstrate the system architecture.)*

5. **Run the Application**
   Start the Streamlit web application by running:
   ```cmd
   streamlit run app.py
   ```

6. **Using the App**
   - The Streamlit interface will open in your web browser.
   - Check the **"Start Webcam"** box in the sidebar.
   - Show your hand to the webcam. MediaPipe will detect your hand landmarks and crop the image to send to the CNN.
   - The prediction will appear on the screen, and the system will read the text out loud.

## Advantages

- Works in real time
- Helps deaf and mute people
- Improves accessibility
- Uses AI and Deep Learning
- Is low-cost and user-friendly

## Conclusion

In conclusion, our project demonstrates how Deep Learning and Computer Vision can be used to build an intelligent gesture recognition system that improves communication and accessibility.
