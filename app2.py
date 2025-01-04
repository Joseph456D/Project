import streamlit as st
import cv2
import time
from ultralytics import YOLO
import pywhatkit
import numpy as np
import threading
import queue

# Load YOLO model
model = YOLO("yolov8n.pt")

# Helper function for fall detection based on bounding box aspect ratio
def is_falling(bbox, aspect_ratio_threshold=0.7):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    aspect_ratio = h / w
    return aspect_ratio < aspect_ratio_threshold

# Function to generate video frames and detect falls
def generate_frames(camera, frame_queue):
    prev_frame_time = 0
    new_frame_time = 0
    fall_alert_sent = False
    last_alert_time = 0  # Track the time of the last alert

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Reduce the resolution for faster processing
        frame = cv2.resize(frame, (640, 480))

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)

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
                        if not fall_alert_sent and (current_time - last_alert_time > 20):  # 20 seconds cooldown
                            try:
                                # Async WhatsApp alert sending
                                pywhatkit.sendwhatmsg_instantly(
                                    phone_no="+91xxxxxxxxxx",  # Replace with actual number
                                    message="Fall Detected!",
                                    tab_close=True
                                )
                                fall_alert_sent = True
                                last_alert_time = current_time
                            except Exception as e:
                                print(f"Error sending WhatsApp message: {e}")
                    else:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Display FPS on the frame
        cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)

        # Convert frame to RGB (for Streamlit compatibility)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Store frame in queue for later use
        frame_queue.put((frame_rgb, fall_detected_in_frame))

# Async frame capture and processing using threading
def capture_video(camera, frame_queue):
    generate_frames(camera, frame_queue)

# Streamlit App
st.set_page_config(page_title="Fall Detection System", page_icon="🏃", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .status-text {
            font-size: 24px;
            font-weight: bold;
            color: #e74c3c;
            text-align: center;
            margin-top: 20px;
        }
        .video-container {
            text-align: center;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 16px;
            color: #7f8c8d;
        }
    </style>
""", unsafe_allow_html=True)

# Header and Instructions
st.markdown('<div class="title">Fall Detection System 🛑</div>', unsafe_allow_html=True)
st.markdown("""
    This is a real-time fall detection system using computer vision. 
    It detects falls from a live video feed using YOLO (You Only Look Once) object detection. 
    If a fall is detected, a WhatsApp message is sent automatically to the designated phone number.
""")

# Video Feed Section
video_placeholder = st.empty()  # Placeholder for video stream
status_placeholder = st.empty()  # Placeholder for status text

fall_detected = False

# Create a queue to hold the frames for real-time display
frame_queue = queue.Queue(maxsize=1)

# Set up camera (with lower resolution to improve performance)
camera = cv2.VideoCapture(0)
camera.set(3, 640)  # Set the width to 640
camera.set(4, 480)  # Set the height to 480

# Start the video capture thread
thread = threading.Thread(target=capture_video, args=(camera, frame_queue), daemon=True)
thread.start()

# Main loop to continuously display video feed and fall detection status
while True:
    if not frame_queue.empty():
        frame, fall_detected_in_frame = frame_queue.get()

        # Update the fall detection status
        if fall_detected_in_frame:
            status_placeholder.markdown('<div class="status-text" style="color: #e74c3c;">**Fall Detected! 🚨**</div>', unsafe_allow_html=True)
        else:
            status_placeholder.markdown('<div class="status-text" style="color: #2ecc71;">No Fall Detected ✅</div>', unsafe_allow_html=True)

        # Display the frame in Streamlit
        video_placeholder.image(frame, caption='Fall Detection Video Feed', use_container_width=True)

    # Allow some time for Streamlit to render the changes
    time.sleep(0.1)

# Footer with copyright or additional details (placed outside the loop)
st.markdown('<div class="footer">Developed with ❤️ by Your Team</div>', unsafe_allow_html=True)