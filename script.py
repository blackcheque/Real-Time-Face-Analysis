import cv2
from deepface import DeepFace
import threading

# Load Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Global variables to share between threads
frame = None
age = None
emotion = None
lock = threading.Lock()

# Function to perform DeepFace analysis asynchronously
def analyze_face(face_roi):
    global age, emotion
    try:
        # Run DeepFace analysis
        analysis = DeepFace.analyze(face_roi, actions=['age', 'emotion'], enforce_detection=False)

        # Access the first result (since analyze returns a list of dictionaries)
        with lock:
            age = analysis[0]['age']
            emotion = analysis[0]['dominant_emotion']
    except Exception as e:
        print(f"DeepFace Error: {e}")

# Process every 5th frame for analysis
frame_skip = 5
frame_count = 0
analysis_thread = None

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Analyze face only every 5th frame
    if len(faces) > 0 and frame_count % frame_skip == 0:
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]  # Extract face region

            # Resize face ROI to speed up DeepFace processing
            small_face_roi = cv2.resize(face_roi, (128, 128))

            # Run DeepFace analysis in a separate thread if it's not already running
            if analysis_thread is None or not analysis_thread.is_alive():
                analysis_thread = threading.Thread(target=analyze_face, args=(small_face_roi,))
                analysis_thread.start()

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # If DeepFace analysis results are available, display them
            with lock:
                if age and emotion:
                    cv2.putText(frame, f"Age: {age}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Real-Time Age and Emotion Detection', frame)

    frame_count += 1  # Increment frame count

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
