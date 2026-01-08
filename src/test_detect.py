import cv2
import numpy as np
import tensorflow as tf
import time

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Open webcam (change to 0 if needed)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Constants
CONFIDENCE_THRESHOLD = 0.7
EDGE_DENSITY_THRESHOLD = 0.01  # Lower = flatter wall

frame_counter = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame early for performance
    frame_resized = cv2.resize(frame, (240, 240))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Detect edges and compute edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Inference only if scene is flat enough
    if edge_density < EDGE_DENSITY_THRESHOLD:
        # Convert grayscale to 3-channel image by repeating grayscale data across 3 channels
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Resize to model input size
        resized_for_model = cv2.resize(gray_3channel, (120, 120))
        
        # Normalize and prepare input tensor (shape: (1, 120, 120, 3))
        input_tensor = np.expand_dims(resized_for_model, axis=0) / 255.0

        prediction = model.predict(input_tensor, verbose=0)[0][0]
        not_cracked_conf = prediction
        cracked_conf = 1.0 - prediction

        # Determine status
        if cracked_conf >= CONFIDENCE_THRESHOLD:
            status = "Positive"
            status_color = (0, 0, 255)
        elif not_cracked_conf >= CONFIDENCE_THRESHOLD:
            status = "Negative"
            status_color = (0, 255, 0)
        else:
            status = "Uncertain"
            status_color = (0, 255, 255)

        # Overlay info
        cv2.putText(frame_resized, f"Cracked: {cracked_conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_resized, f"Not Cracked: {not_cracked_conf:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_resized, f"Status: {status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    else:
        cv2.putText(frame_resized, "No Wall Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Calculate and overlay FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show only every 2nd frame to boost GUI smoothness
    if frame_counter % 2 == 0:
        cv2.imshow("Live Crack Detection", frame_resized)
    frame_counter += 1

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
