import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('spinach_leaf_model.h5')  # Update with your model file

# Define class labels
class_labels = {
    0: "Anthracnose",
    1: "Bacterial Spot",
    2: "Downy Mildew",
    3: "Healthy Leaf",
    4: "Pest Damage"
}

# IP Camera URL 
ip_camera_url = "Replace with your IP camera feed URL" 

# RTSP Output Stream URL (Change this based on your RTSP server settings)
rtsp_output_url = "rtsp://192.168.1.10:8554/live"  # Example: Use a GStreamer/FFmpeg RTSP server

# Start video capture
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Could not open IP camera.")
    exit()

# Define output video writer for RTSP stream
frame_width, frame_height = 1280, 720  # Set resolution to 720x480
fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use H.264 codec for streaming
out = cv2.VideoWriter(rtsp_output_url, fourcc, 30.0, (frame_width, frame_height))

# Function to process and predict an image
def predict_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to model input size
    img_array = img_to_array(frame_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get class with highest probability
    confidence = predictions[0][predicted_class]  # Confidence score

    return class_labels[predicted_class], confidence

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame for display and RTSP output
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Predict class for the current frame
    predicted_label, confidence = predict_frame(frame)

    # Display prediction on the frame
    text = f"{predicted_label} ({confidence:.2f})"
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write frame to RTSP stream
    out.write(frame)

    # Show the video feed with predictions
    cv2.imshow("Live Prediction", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
