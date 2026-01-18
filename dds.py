from scipy.spatial import distance as dist
from tkinter import *
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import cv2
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[ERROR] MediaPipe not installed. Run: pip install mediapipe")


# -----------------------------
# Robust file paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(SCRIPT_DIR, "alarm.wav")


# -----------------------------
# Tkinter main window
# -----------------------------
root = Tk()
root.geometry("400x300")
root.configure(bg="black")
root.maxsize(400, 300)
root.title("Drowsiness Detection System")


# -----------------------------
# Helpers
# -----------------------------
def sound_alarm():
    try:
        playsound.playsound(ALARM_PATH)
    except Exception as e:
        print(f"[WARN] alarm sound failed: {e}")


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# -----------------------------
# Core logic
# -----------------------------
def start(EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, webcam_index=0):
    if not MEDIAPIPE_AVAILABLE:
        print("[ERROR] MediaPipe is required. Install it with: pip install mediapipe")
        return

    if not os.path.exists(ALARM_PATH):
        print("[WARN] Missing alarm.wav in:", SCRIPT_DIR)

    print("[INFO] loading MediaPipe face mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # MediaPipe eye landmark indices
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    COUNTER = 0
    ALARM_ON = False

    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=webcam_index).start()
    time.sleep(1.0)

    i = 0
    min_ear = 999.0
    max_ear = 0.0
    ear = 0.0

    while True:
        frame = vs.read()

        # camera returns nothing -> skip safely
        if frame is None:
            print("[WARN] frame is None (camera permission/index?).")
            time.sleep(0.05)
            continue

        # Resize frame
        frame = imutils.resize(frame, width=650)
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show settings on the display frame
        cv2.putText(frame, f"EYE_AR_THRESH = {EYE_AR_THRESH}", (10, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"EYE_AR_CONSEC_FRAMES = {EYE_AR_CONSEC_FRAMES}", (300, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Process frame with MediaPipe
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks
                left_eye_coords = []
                for idx in LEFT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    left_eye_coords.append([x, y])
                
                right_eye_coords = []
                for idx in RIGHT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    right_eye_coords.append([x, y])

                left_eye_coords = np.array(left_eye_coords)
                right_eye_coords = np.array(right_eye_coords)

                # Calculate EAR for both eyes
                leftEAR = eye_aspect_ratio(left_eye_coords)
                rightEAR = eye_aspect_ratio(right_eye_coords)
                ear = (leftEAR + rightEAR) / 2.0

                # Draw eye contours
                leftEyeHull = cv2.convexHull(left_eye_coords)
                rightEyeHull = cv2.convexHull(right_eye_coords)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True
                            t = Thread(target=sound_alarm, daemon=True)
                            t.start()

                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    ALARM_ON = False

                cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # press q to quit
        if key == ord("q"):
            break

        # optional auto threshold calibration
        if i < 50:
            if ear > 0:  # Only update if we detected a face
                min_ear = min(min_ear, ear)
                max_ear = max(max_ear, ear)
            i += 1
        elif i == 50:
            if max_ear > 0:
                EYE_AR_THRESH = (min_ear + max_ear) / 2.0
                print(f"[INFO] Auto-calibrated EYE_AR_THRESH to {EYE_AR_THRESH:.3f}")
            i += 1

    cv2.destroyAllWindows()
    vs.stop()
    face_mesh.close()


# -----------------------------
# Settings UI
# -----------------------------
def settings():
    settingsBtn.pack_forget()
    startBtn.pack_forget()

    EATList = ['0.25', '0.26', '0.27', '0.28', '0.29', '0.30', '0.31', '0.32', '0.33', '0.34', '0.35']
    EACList = [30, 35, 40, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 60]
    CAMList = [0, 1, 2]

    v1 = StringVar(value='0.31')
    v2 = IntVar(value=48)
    v3 = IntVar(value=0)

    Label(root, text="Set Threshold EAR:", bg="black", fg="white").pack(pady=5)
    opt1 = OptionMenu(root, v1, *EATList)
    opt1.config(width=20, font=('Helvetica', 12))
    opt1.pack()

    Label(root, text="Set consecutive frames:", bg="black", fg="white").pack(pady=5)
    opt2 = OptionMenu(root, v2, *EACList)
    opt2.config(width=20, font=('Helvetica', 12))
    opt2.pack()

    Label(root, text="Camera index (try 0 or 1):", bg="black", fg="white").pack(pady=5)
    opt3 = OptionMenu(root, v3, *CAMList)
    opt3.config(width=20, font=('Helvetica', 12))
    opt3.pack()

    Button(
        root,
        text="SAVE AND START",
        command=lambda: start(float(v1.get()), v2.get(), v3.get()),
        bg="black",
        fg="white",
    ).pack(pady=25)


# -----------------------------
# Buttons
# -----------------------------
settingsBtn = Button(root, text="SETTINGS", command=settings, bg="black", fg="white")
settingsBtn.pack(pady=55)

startBtn = Button(root, text="START", command=lambda: start(0.31, 48, 0), bg="black", fg="white")
startBtn.pack(pady=35)

root.mainloop()