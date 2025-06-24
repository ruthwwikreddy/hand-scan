from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
import threading
from queue import Queue
import os

app = Flask(__name__)

# Load hand template image
original_template = cv2.imread('static/hand_template.png', cv2.IMREAD_UNCHANGED)
if original_template is None:
    raise FileNotFoundError("hand_template.png not found in static/ folder.")

# Scale the template to make it larger (1.5x)
scale_factor = 1.5
template = cv2.resize(original_template, None, fx=scale_factor, fy=scale_factor)
template_h, template_w = template.shape[:2]

# Queue for frames
frame_queue = Queue(maxsize=10)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# For controlling video trigger
video_triggered = False
hand_was_outside = True

def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    h, w = img_overlay.shape[:2]

    if y + h > img.shape[0] or x + w > img.shape[1]:
        return

    alpha_overlay = img_overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):
        img[y:y+h, x:x+w, c] = (
            alpha_overlay * img_overlay[:, :, c] +
            alpha_background * img[y:y+h, x:x+w, c]
        )

def process_camera():
    global video_triggered, hand_was_outside
    cap = None
    video = None
    video_playing = False

    while True:
        # Open camera only if video is not playing
        if not video_playing:
            if cap is None:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Error: Could not open webcam")
                    break
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if cap is not None:
                cap.release()
                cap = None

        if not video_playing:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2
            overlay_image_alpha(frame, template, (center_x, center_y))

            # Hand detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_in_position = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

                    if (center_x < x_min < center_x + template_w and
                        center_y < y_min < center_y + template_h and
                        center_x < x_max < center_x + template_w and
                        center_y < y_max < center_y + template_h):
                        hand_in_position = True
                        cv2.putText(frame, '✅ Hand in correct position!',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if hand_in_position and hand_was_outside:
                hand_was_outside = False
                video_triggered = True
                video_playing = True
                # Open video file
                video = cv2.VideoCapture('static/vid.mp4')
                if not video.isOpened():
                    print("Error: Could not open video file")
                    video = None
                    video_playing = False
                    video_triggered = False

        if video_playing and video is not None:
            ret, video_frame = video.read()
            if ret:
                # Resize video frame to match camera feed
                video_frame = cv2.resize(video_frame, (w, h))
                # Overlay video on camera feed
                alpha = 0.7  # Adjust transparency (0.0 to 1.0)
                frame = cv2.addWeighted(frame, 1 - alpha, video_frame, alpha, 0)
            else:
                # Video ended, restart it from the beginning
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to frame 0
                ret, video_frame = video.read()  # Read first frame
                if ret:
                    # Resize video frame to match camera feed
                    video_frame = cv2.resize(video_frame, (w, h))
                    # Overlay video on camera feed
                    alpha = 0.7  # Adjust transparency (0.0 to 1.0)
                    frame = cv2.addWeighted(frame, 1 - alpha, video_frame, alpha, 0)
                else:
                    print("Error: Could not read video frame after restart")
                    video.release()
                    video = None
                    video_playing = False
                    video_triggered = False
                    hand_was_outside = True

        _, buffer = cv2.imencode('.jpg', frame)
        if buffer is not None and frame_queue.qsize() < 10:
            frame_queue.put(buffer.tobytes())

    if cap is not None:
        cap.release()
    if video is not None:
        video.release()

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2
        overlay_image_alpha(frame, template, (center_x, center_y))

        # Hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_in_position = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

                if (center_x < x_min < center_x + template_w and
                    center_y < y_min < center_y + template_h and
                    center_x < x_max < center_x + template_w and
                    center_y < y_max < center_y + template_h):
                    hand_in_position = True
                    cv2.putText(frame, '✅ Hand in correct position!',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if hand_in_position and hand_was_outside:
            hand_was_outside = False
            video_triggered = True
            # Once video is triggered, keep it playing until it ends
            hand_was_outside = False
            # Open video file
            video = cv2.VideoCapture('static/vid.mp4')
            if not video.isOpened():
                print("Error: Could not open video file")
                video = None

        if video_triggered and video is not None:
            ret, video_frame = video.read()
            if ret:
                # Resize video frame to match camera feed
                video_frame = cv2.resize(video_frame, (w, h))
                # Overlay video on camera feed
                alpha = 0.7  # Adjust transparency (0.0 to 1.0)
                frame = cv2.addWeighted(frame, 1 - alpha, video_frame, alpha, 0)

        _, buffer = cv2.imencode('.jpg', frame)
        if buffer is not None and frame_queue.qsize() < 10:
            frame_queue.put(buffer.tobytes())

    if cap is not None:
        cap.release()
    if video is not None:
        video.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return {'video_triggered': video_triggered}

if __name__ == '__main__':
    if not os.path.exists('static/vid.mp4'):
        print("Error: vid.mp4 not found in static/ folder. Please place the video file there.")
        exit(1)
    
    threading.Thread(target=process_camera, daemon=True).start()
    app.run(host='0.0.0.0', port=5050, debug=False)
