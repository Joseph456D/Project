import app as st
import cv2
import time
import threading
from ultralytics import YOLO
import pywhatkit
import numpy as np

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
def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return None, False

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
                        if not fall_alert_sent and (current_time - last_alert_time > 20):  # 20 seconds cooldown
                            try:
                                pywhatkit.sendwhatmsg_instantly(
                                    phone_no="+91xxxxxxxxxx",  # Replace with actual number
                                    message="Fall Detected!",
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

                # Display FPS on the frame
                cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)

        # Convert frame to RGB (for Streamlit compatibility)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Return frame and fall detection status
        return frame_rgb, fall_detected_in_frame

# Streamlit App
st.title("Fall Detection System")

# Video Feed Section
video_placeholder = st.empty()  # Placeholder for video stream
status_placeholder = st.empty()  # Placeholder for status text

fall_detected = False

# Main loop to continuously display video feed and fall detection status
while True:
    frame, fall_detected_in_frame = generate_frames()

    # Update the fall detection status
    if fall_detected_in_frame:
        status_placeholder.markdown("**Fall Detected!**")
    else:
        status_placeholder.markdown("No Fall Detected")

    # Display the frame in Streamlit
    video_placeholder.image(frame, caption='Fall Detection Video Feed', use_container_width=True)

    # Allow some time for Streamlit to render the changes
    time.sleep(0.1)
