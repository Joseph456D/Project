import streamlit as st
import cv2
import time
from ultralytics import YOLO
import pywhatkit
import numpy as np
import threading
import queue
import re  # For validating phone numbers

# Load YOLO model
model = YOLO("yolov8n.pt")

# Function to validate phone number (basic validation, can be expanded)
def validate_phone_number(phone_number):
    phone_regex = re.compile(r'^\+?\d{10,15}$')
    return bool(phone_regex.match(phone_number))

# Helper function for fall detection based on bounding box aspect ratio
def is_falling(bbox, aspect_ratio_threshold=0.7):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    aspect_ratio = h / w
    return aspect_ratio < aspect_ratio_threshold

# Function to generate video frames and detect falls
def generate_frames(camera, frame_queue, phone_numbers):
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
                                # Async WhatsApp alert sending to multiple phone numbers
                                for phone_number in phone_numbers:
                                    pywhatkit.sendwhatmsg_instantly(
                                        phone_no=phone_number,  # Use the current phone number
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

        # Convert frame to RGB (for Streamlit compatibility)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Store frame in queue for later use
        frame_queue.put((frame_rgb, fall_detected_in_frame))

# Async frame capture and processing using threading
def capture_video(camera, frame_queue, phone_numbers):
    generate_frames(camera, frame_queue, phone_numbers)

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

# Sidebar Configuration for Camera and Phone Number Selection
st.sidebar.title("Settings")

# Move camera source option to the top
camera_source = st.sidebar.selectbox(
    "Select Camera Source",
    options=["Local Camera (Webcam)", "RTSP Camera"]
)

# Check if initial emergency contact number is already in session state
if 'phone_numbers' not in st.session_state:
    # Show pop-up-like behavior to set the initial phone number
    initial_contact_number = st.sidebar.text_input(
        "Enter Initial Emergency Contact Number",
        placeholder="+91xxxxxxxxxx",  # default value or pre-populated number
        help="Enter the phone number to send initial alerts. You can modify it later."
    )
    
    # When the user enters the phone number, save it to session state
    if initial_contact_number:
        initial_contact_number = initial_contact_number.strip()
        if initial_contact_number and validate_phone_number(initial_contact_number):
            st.session_state['phone_numbers'] = [initial_contact_number]
        else:
            st.sidebar.warning("Invalid phone number format. Please use a valid number.")

# If phone numbers are already set, proceed to camera settings
phone_numbers = st.session_state.get('phone_numbers', [])

if not phone_numbers:
    st.warning("Please set at least one phone number before starting fall detection.")

# Button to add additional phone numbers
if st.sidebar.button("Add Additional Phone Number"):
    with st.sidebar:
        additional_phone_number = st.text_input(
            "Enter Additional Emergency Contact Number",
            value=""
        )
        if additional_phone_number:
            additional_phone_number = additional_phone_number.strip()
            if validate_phone_number(additional_phone_number):
                if additional_phone_number not in phone_numbers:
                    phone_numbers.append(additional_phone_number)
                    st.session_state['phone_numbers'] = phone_numbers  # Update session state with new number
                else:
                    st.sidebar.warning("This phone number has already been added.")
            else:
                st.sidebar.warning("Invalid phone number format. Please use a valid number.")

# Button to show/hide all added phone numbers
if 'show_numbers' not in st.session_state:
    st.session_state['show_numbers'] = False

if st.sidebar.button("Show/Hide Added Numbers"):
    st.session_state['show_numbers'] = not st.session_state['show_numbers']

if st.session_state['show_numbers']:
    st.sidebar.write("Added Phone Numbers:")
    for number in phone_numbers:
        st.sidebar.write(number)
else:
    st.sidebar.empty()
    for number in phone_numbers:
        st.sidebar.empty()

# Video Feed Section
if phone_numbers:
    st.markdown('<div class="title">Fall Detection System 🛑</div>', unsafe_allow_html=True)
    st.markdown("""
        This is a real-time fall detection system using computer vision. 
        It detects falls from a live video feed using YOLO (You Only Look Once) object detection. 
        If a fall is detected, a WhatsApp message is sent automatically to the designated phone numbers.
    """)

    # Video Feed Placeholder
    video_placeholder = st.empty()  # Placeholder for video stream
    status_placeholder = st.empty()  # Placeholder for status text

    fall_detected = False

    # Create a queue to hold the frames for real-time display
    frame_queue = queue.Queue(maxsize=1)

    rtsp_url = ""
    if camera_source == "RTSP Camera":
        rtsp_url = st.sidebar.text_input("Enter RTSP Camera URL", "rtsp://username:password@ip_address:port/stream_path")
    else:
        st.sidebar.empty()

    # Set up camera (with lower resolution to improve performance)
    if camera_source == "RTSP Camera" and rtsp_url:
        camera = cv2.VideoCapture(rtsp_url)  # Open RTSP stream
        if not camera.isOpened():
            st.error("Failed to connect to the RTSP stream.")
    else:
        camera = cv2.VideoCapture(0)  # Use local webcam
        camera.set(3, 640)  # Set the width to 640
        camera.set(4, 480)  # Set the height to 480

    # Start the video capture thread
    thread = threading.Thread(target=capture_video, args=(camera, frame_queue, phone_numbers), daemon=True)
    thread.start()

    # Footer with copyright or additional details (placed outside the loop)
    st.markdown('<div class="footer">Developed with Streamlit & YOLOv7 by Group 7</div>', unsafe_allow_html=True)

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
