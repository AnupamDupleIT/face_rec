import cv2
import os

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haar_cascades/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

# Define the count variable to keep track of the number of images captured
count = 0

# Get the name of the person to save the images before starting the loop
name = input("Enter the name of the person: ")

# Create a folder to save the images
if not os.path.exists("faces/" + name):
    os.makedirs("faces/" + name)

while True:
    # Read the frames from the webcam
    ret, frame = cap.read()

    # Convert the frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Save the captured images
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        cv2.imwrite("faces/" + name + "/" + name + str(count) + ".jpg", roi_gray)
        count += 1
    
    if count >= 300:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
