import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

def create_dummy_model():
    print("Creating a CNN model architecture as described in the project...")
    
    model = Sequential([
        # Input layer: 64x64 pixels, 1 channel (Grayscale)
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output layer: 5 Gestures (Hello, Yes, No, Thanks, Help)
        Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.summary()
    
    # Save the untrained model as a dummy for the Streamlit app to load
    model_path = 'gesture_model.h5'
    model.save(model_path)
    print(f"\nModel successfully saved to {os.path.abspath(model_path)}")
    print("Note: This is an untrained dummy model for demonstration.")
    print("To make it accurate, you will need to train it with actual dataset images.")

if __name__ == "__main__":
    create_dummy_model()
