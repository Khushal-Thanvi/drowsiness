from flask import Flask, render_template, Response
import cv2
import time
from playsound import playsound
import threading
import os

# Set DirectShow as preferred backend on Windows
if os.name == 'nt':
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

app = Flask(__name__)

# Variables for drowsiness detection
blink_counter = 0
eyes_closed_start = None
drowsy_detected = False
alert_cooldown = False
BLINK_THRESHOLD = 0.5  # Time in seconds for eyes to be closed to count as a blink
DROWSY_THRESHOLD = 2.0  # Time in seconds for eyes to be closed to trigger drowsiness
ALERT_COOLDOWN_TIME = 5.0  # Time in seconds before another alert can be triggered

print("[INFO] Loading Haar cascade classifier...")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
if eye_cascade.empty():
    print("[ERROR] Failed to load haarcascade_eye.xml. Please check the file path.")
    exit(1)

def initialize_camera():
    print("[INFO] Initializing camera with DirectShow backend...")
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Failed to access the camera.")
        return None
    return cap

def play_alert():
    global alert_cooldown
    try:
        playsound('sound.wav')
        alert_cooldown = True
        threading.Timer(ALERT_COOLDOWN_TIME, reset_alert_cooldown).start()
    except Exception as e:
        print(f"[ERROR] Failed to play alert: {str(e)}")

def reset_alert_cooldown():
    global alert_cooldown
    alert_cooldown = False

def generate_frames():
    global blink_counter, eyes_closed_start, drowsy_detected, alert_cooldown
    
    camera = initialize_camera()
    if camera is None:
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Convert to grayscale and detect eyes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_time = time.time()
        
        # Process eye detection results
        if len(eyes) == 0:  # No eyes detected (closed or not visible)
            if eyes_closed_start is None:
                eyes_closed_start = current_time
            else:
                eyes_closed_duration = current_time - eyes_closed_start
                
                # Check for blink
                if eyes_closed_duration >= BLINK_THRESHOLD:
                    if not drowsy_detected:  # Only increment if not already drowsy
                        blink_counter += 1
                
                # Check for drowsiness
                if eyes_closed_duration >= DROWSY_THRESHOLD:
                    drowsy_detected = True
                    if not alert_cooldown:
                        threading.Thread(target=play_alert).start()
        else:
            # Eyes detected (open)
            eyes_closed_start = None
            if drowsy_detected:
                drowsy_detected = False
                blink_counter = 0
            
            # Draw rectangles around detected eyes
            for (x, y, w, h) in eyes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add status information to frame
        status_color = (0, 0, 255) if drowsy_detected else (0, 255, 0)
        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if drowsy_detected:
            cv2.putText(frame, "DROWSINESS DETECTED!", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add alert cooldown indicator
        if alert_cooldown:
            cv2.putText(frame, "Alert Cooldown Active", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)