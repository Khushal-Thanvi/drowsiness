import cv2
import time
from playsound import playsound

# Variables to track blinking frequency
blink_counter = 0
blink_start_time = None

# Load the pre-trained Haar cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Capture video feed from the webcam or external camera
cap = cv2.VideoCapture(0)

dt = False
cs = 24 * 3

while True:
    # Read the current frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes in the grayscale frame
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in eyes:
        # Draw rectangles around the detected eyes
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Measure the time duration between consecutive blinks
    if len(eyes) == 0:
        if blink_start_time is None:
            blink_start_time = time.time()
        else:
            if time.time() - blink_start_time > 0.3:
                blink_counter += 1
                blink_start_time = None
    else:
        blink_counter = 0
        blink_start_time = None
    
    # In case of drowsiness, inform the driver.
    if blink_counter >= 5 or dt == True:
        blink_counter = 0
        cv2.putText(frame, "Drowsiness Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        dt = True
        cs -= 1
        if cs == 0:
            playsound('sound.wav')
            dt = False
            cs = 24 * 3
    
    # Display the frame with eye rectangles and blinking frequency
    cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Drowsiness Detection', frame)
    
    # If 'q' is pushed, the loop will end.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video recording, then shut the window.
cap.release()
cv2.destroyAllWindows()