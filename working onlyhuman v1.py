from flask import Flask, render_template_string, Response, jsonify
import cv2
import threading
import time
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()
model = YOLO("yolov8n.pt")

# HTML Template with improved styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fall Detection System</title>
    <style>
        body {
            margin: 0;
            font-family: 'Arial', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            background-color: #f5f5f5;
            min-height: 100vh;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        #video_feed {
            width: 640px;
            height: 480px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
            border: 5px solid #3498db;
        }
        .alert {
            color: red;
            font-weight: bold;
            font-size: 18px;
            margin-top: 20px;
            text-align: center;
        }
    </style>
    <script>
        let fallAlerted = false;
        let lastFallTime = 0;

        function checkFall(fallDetected) {
            const currentTime = Date.now();
            if (fallDetected && (currentTime - lastFallTime > 5000)) { // 5-second cooldown
                alert("FALL DETECTED!");
                lastFallTime = currentTime;
            }
        }

        setInterval(async () => {
            const response = await fetch('/fall_status');
            const data = await response.json();
            checkFall(data.fallDetected);
        }, 500); // Check fall status every 0.5 seconds
    </script>
</head>
<body>
    <h1>Fall Detection System</h1>
    <img id="video_feed" src="{{ url_for('video_feed') }}" alt="Video Feed">
    <p class="alert">Waiting for Fall Detection...</p>
</body>
</html>
"""

# Fall detection function using YOLO model
def generate_frames():
    global latest_frame, frame_lock
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)

        fall_detected_in_frame = False  # Check if fall is detected in the current frame
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                aspect_ratio = h / w

                # Only detect humans (class label 0 in YOLO)
                if box.cls == 0:  # Human class
                    # More accurate fall detection based on aspect ratio
                    if aspect_ratio < 0.7:
                        fall_detected_in_frame = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, "Fall Detected!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Lock frame update to ensure thread safety
        with frame_lock:
            latest_frame = (frame, fall_detected_in_frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes
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

# Run Flask and fall detection in parallel
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
