import cv2     #Open Source Computer Vision library 
import mediapipe as mp
import fitz
import numpy as np
import time
import threading
from collections import deque

def highlight_word(img, words, index, color=(0, 0, 255), thickness=3):
    """Draw highlight on a specific word index."""
    frame = img.copy()
    if 0 <= index < len(words):
        x0, y0, x1, y1, text, _, _, _ = words[index]
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
    return frame

def process_camera(shared):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(    
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    prev_gaze = None
    while not shared['stop']:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        gaze = "CENTER"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = face_landmarks.landmark

                right_right_corner = landmarks[33]
                right_left_corner = landmarks[133]

                left_left_corner = landmarks[263]
                left_right_corner = landmarks[362]

                r_lx, r_ly = int(right_left_corner.x*w), int(right_left_corner.y*h)
                r_rx, r_ry = int(right_right_corner.x*w), int(right_right_corner.y*h)

                l_lx, l_ly = int(left_left_corner.x*w), int(left_left_corner.y*h)
                l_rx, l_ry = int(left_right_corner.x*w), int(left_right_corner.y*h)

                right_iris = [landmarks[i] for i in [474, 475, 476, 477]]
                right_center = landmarks[473]
                right_x, right_y = int(right_center.x*w), int(right_center.y*h)

                left_iris = [landmarks[i] for i in [469, 470, 471, 472]]
                left_center = landmarks[468]
                left_x, left_y = int(left_center.x*w), int(left_center.y*h)

                right_ratio = (right_x-r_lx)/(r_rx-r_lx)
                left_ratio = (left_x-l_lx)/(l_rx-l_lx)
                avg_ratio = (left_ratio + right_ratio)/2

                print("Average ratio:", round(avg_ratio, 3))

                ratios = deque(maxlen=5)
                ratios.append(avg_ratio)
                smoothed = sum(ratios)/len(ratios)

                if smoothed < 0.445:
                    gaze = "LEFT"
                
                elif smoothed > 0.515:
                    gaze = "RIGHT"

                else:
                    gaze = "CENTER"

                if gaze!=prev_gaze:
                    shared["gaze"] = gaze
                    prev_gaze = gaze


                # cv2.circle(frame, (right_x, right_y), 3, (0, 0, 255), -1)
                # cv2.circle(frame, (left_x, left_y), 3, (255, 0, 0), -1)

                # print(f"Left Iris: ({left_x}, {left_y})  Right Iris: ({right_x}, {right_y})")

            flipped = cv2.flip(frame, 1)
            cv2.putText(flipped, f"Gaze: {gaze}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam Face & Eyes", flipped)

        if cv2.waitKey(1) & 0xFF == 27:
            shared["stop"]=True
            break

    cap.release()
    cv2.destroyAllWindows()

def display_pdf(shared):
    # Open PDF and get first page
    doc = fitz.open(r"C:\Users\Kartik\OneDrive\Desktop\study\CSE\DSA\PDF_eye_tracking_project\test.pdf")
    page = doc[0]

    pix = page.get_pixmap()

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)

    # Extract words
    words = page.get_text("words")

    index = 0
    # delay = 500  # milliseconds between words

    prev_gaze = None

    while not shared["stop"]:
        gaze = shared["gaze"]
        if gaze != prev_gaze:
            if gaze == "RIGHT" and index < len(words) - 1:
                index += 1
            elif gaze == "LEFT" and index > 0:
                index -= 1
            prev_gaze = gaze

        frame = highlight_word(img, words, index)
        cv2.imshow("PDF Words Highlight", frame)

        key = cv2.waitKey(200) & 0xFF
        if key == 27:
            shared["stop"]=True
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    shared_data = {"gaze":"CENTER", "index":0, "stop":False}

    t1 = threading.Thread(target=process_camera, args=(shared_data,))
    t2 = threading.Thread(target=display_pdf, args=(shared_data,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Exited cleanly.")

        







