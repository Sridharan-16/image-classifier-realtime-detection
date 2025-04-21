import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model(r"ADD_YOUR_.h5_path")

# Define class names (Modify based on your dataset)
class_names = ["IN_THIS_GIVE_HOW_MANY_CLASSES_YOU_HAVE"]  # Replace with your actual class names

# Set image size based on training
img_size = (180, 180)

def preprocess_frame(frame):
    """Preprocess the frame before feeding it to the model."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to PIL format
    img = img.resize(img_size)  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()  # Capture frame
    if not ret:
        break

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection

    # Find contours (potential objects)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore small detections (noise filtering)
        if w < 50 or h < 50:
            continue

        # Extract the object from the frame
        object_roi = frame[y:y+h, x:x+w]

        # Preprocess the extracted object
        img_array = preprocess_frame(object_roi)

        # Predict the object class
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Draw bounding box if confidence is high
        if confidence > 0.90:
            label = f"{predicted_class} ({confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with bounding boxes
    cv2.imshow("Object Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
