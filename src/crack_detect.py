import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Open webcam (change to 0 for turtlebot)
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Thresholds
CONFIDENCE_THRESHOLD = 0.7
EDGE_DENSITY_THRESHOLD = 0.01  # Lower = flatter wall

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Compute edge density
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = edge_pixels / total_pixels

    if edge_density < EDGE_DENSITY_THRESHOLD:
        # Resize input for prediction
        resized_frame = cv2.resize(frame, (120, 120))
        input_tensor = np.expand_dims(resized_frame, axis=0) / 255.0

        prediction = model.predict(input_tensor, verbose=0)[0][0]
        not_cracked_conf = prediction
        cracked_conf = 1.0 - prediction

        # Determine label
        if cracked_conf >= CONFIDENCE_THRESHOLD:
            status = "Positive"
            status_color = (0, 0, 255)
        elif not_cracked_conf >= CONFIDENCE_THRESHOLD:
            status = "Negative"
            status_color = (0, 255, 0)
        else:
            status = "Uncertain"
            status_color = (0, 255, 255)

        # Display both confidence values
        cv2.putText(frame, f"Cracked: {cracked_conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Not Cracked: {not_cracked_conf:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show final status
        cv2.putText(frame, f"Status: {status}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
    else:
        # Too many lines → not a wall
        cv2.putText(frame, "No Wall Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display final frame
    cv2.imshow("Live Crack Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
