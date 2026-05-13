import cv2
import numpy as np
import os

WINDOW_NAME = 'BallTracker'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, 'ball3.mp4')

COLOR_RANGES = {
    'vert':  ([40,  70,  70],  [80,  255, 255]),
    'bleu':  ([100, 100, 50],  [130, 255, 255]),
    'jaune': ([20,  100, 100], [35,  255, 255]),
    'rose':  ([140, 50,  50],  [170, 255, 255]),
}

def track(image, color='vert'):
    lower, upper = COLOR_RANGES[color]
    blur  = cv2.GaussianBlur(image, (5, 5), 0)
    hsv   = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, np.array(lower), np.array(upper))
    bmask = cv2.GaussianBlur(mask, (5, 5), 0)

    moments = cv2.moments(bmask)
    m00 = moments['m00']
    ctr = (-1, -1)
    if m00 != 0:
        cx = int(moments['m10'] / m00)
        cy = int(moments['m01'] / m00)
        ctr = (cx, cy)
        cv2.circle(image, ctr, 10, (0, 0, 125), -1)

    cv2.imshow(WINDOW_NAME, image)
    if cv2.waitKey(1) & 0xFF == 27:
        return None
    return ctr

if __name__ == '__main__':
    color = input("Couleur (vert/bleu/jaune/rose) : ").strip().lower()
    if color not in COLOR_RANGES:
        print("Couleur invalide")
        exit()

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir ball3.mp4")
        print("Chemin actuel :", os.getcwd())
        exit()
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fin de vidéo ou erreur de lecture")
            break
        if track(frame, color) is None:
            break
    cap.release()
    cv2.destroyAllWindows()