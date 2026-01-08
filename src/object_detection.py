import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')
print("✅ Model loaded!")

# Labels for classification (adjust to match your model)
labels = ['Non-Crack', 'Crack']  # <-- Replace with your actual classes

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("🎥 Camera started. Running recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Can't receive frame")
        break

    # Resize to model input shape
    resized_frame = cv2.resize(frame, (224, 224))  # Use your model's input size
    input_data = resized_frame / 255.0  # Normalize if your model was trained this way
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(input_data)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    predicted_label = labels[predicted_index]
    print(f"🔍 Detected: {predicted_label} ({confidence*100:.2f}%)")

    cv2.waitKey(100)  # ~10 FPS

cap.release()
