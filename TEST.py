from flask import Flask, render_template_string, Response, jsonify, url_for
import cv2
import time
import threading
from ultralytics import YOLO
import numpy as np
import pywhatkit
import os

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()
model = YOLO("yolov8n.pt")

# HTML Template for the Fall Detection System
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fall Detection System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <header>
        <h1>Fall Detection System</h1>
    </header>
    <main>
        <section>
            <h2>Live Video Feed</h2>
            <img id="video_feed" src="{{ url_for('video_feed') }}">
        </section>
        <section>
            <h2>Fall Status</h2>
            <p id="fall_status">No Fall Detected</p>
        </section>
    </main>
    <script>
        let lastFallTime = 0;

        async function checkFall() {
            const response = await fetch('/fall_status');
            const data = await response.json();
            const fallStatusElement = document.getElementById('fall_status');
            if (data.fallDetected) {
                const currentTime = Date.now();
                if (currentTime - lastFallTime > 5000) {
                    fallStatusElement.textContent = "FALL DETECTED!";
                    lastFallTime = currentTime;
                }
            } else {
                fallStatusElement.textContent = "No Fall Detected";
            }
        }

        setInterval(checkFall, 500); // Check for fall status every 500ms
    </script>
</body>
</html>
"""

# Helper function for fall detection based on bounding box aspect ratio
def is_falling(bbox, aspect_ratio_threshold=0.7):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    aspect_ratio = h / w
    return aspect_ratio < aspect_ratio_threshold

# Function to generate video frames
def generate_frames():
    global latest_frame, frame_lock
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_frame_time = 0
    new_frame_time = 0
    fall_alert_sent = False
    last_alert_time = 0  # Track the time of the last alert

    while True:
        success, frame = camera.read()
        if not success:
            break

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        # YOLO model tracking
        results = model.track(frame, persist=True, classes=0)
        fall_detected_in_frame = False

        if results and results[0].boxes:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].tolist()
                conf = box.conf.tolist()[0]
                if conf > 0.5:
                    if is_falling(xyxy):
                        fall_detected_in_frame = True
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, "Fall Detected!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        current_time = time.time()
                        if not fall_alert_sent and (current_time - last_alert_time > 20):  # 1-minute cooldown
                            try:
                                pywhatkit.sendwhatmsg_instantly(
                                    phone_no="+917306144159",  # Replace with actual number
                                    message="ninte valiyama veenada myre!",
                                    tab_close=True
                                )
                                print("WhatsApp message sent (using pywhatkit)")
                                fall_alert_sent = True
                                last_alert_time = current_time
                            except Exception as e:
                                print(f"Error sending WhatsApp message: {e}")
                    else:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)

        # Reset the fall_alert_sent flag if no fall is detected
        if not fall_detected_in_frame:
            fall_alert_sent = False

        with frame_lock:
            latest_frame = (frame, fall_detected_in_frame)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fall_status')
def fall_status():
    global latest_frame, frame_lock
    with frame_lock:
        if latest_frame:
            _, fall_detected = latest_frame
            return jsonify({'fallDetected': fall_detected})
        else:
            return jsonify({'fallDetected': False})

if __name__ == '__main__':
    app.static_folder = 'static'
    os.makedirs(app.static_folder, exist_ok=True)

    # Save the CSS file
    with open(os.path.join(app.static_folder, 'style.css'), 'w') as f:
        f.write("""
/* style.css */
body {
  font-family: sans-serif;
  margin: 0;
  background-color: #f4f4f4;
  color: #333;
  line-height: 1.6;
}

header {
  background-color: #007bff;
  color: white;
  padding: 1rem 0;
  text-align: center;
}

main {
  padding: 2rem;
}

section {
  margin-bottom: 2rem;
  background-color: white;
  padding: 1rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

#video_feed {
  width: 640px;
  height: 480px;
  margin: 0 auto;
  display: block;
}

#fall_status {
  font-weight: bold;
  font-size: 1.2rem;
  text-align: center;
  margin-top: 10px;
}
""")
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
