import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model("models/model.h5")
output=['Deepali', 'anupam', 'atul', 'azim', 'deepak', 'dhanesh', 'pankaj', 'prikshit', 'suraj']


# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Get the current frame
    ret, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the Haar cascade classifier to detect faces
    face_cascade = cv2.CascadeClassifier("haar_cascades/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10, minSize=(30, 30))

    # Draw a rectangle around each face and predict the label
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_color,(80,80))
        roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
        roi_gray = roi_gray.reshape(1,80,80)    
        label = model.predict(roi_gray)
        label=output[np.argmax(label)]
        print(label)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Camera", frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
