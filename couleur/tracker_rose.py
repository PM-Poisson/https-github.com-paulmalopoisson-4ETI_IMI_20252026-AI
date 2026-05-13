import cv2
import numpy as np
import os

WINDOW_NAME = 'BallTracker - Rose'

COLOR_RANGES = {
    'vert':  ([40,  70,  70],  [80,  255, 255]),
    'bleu':  ([100, 100, 50],  [130, 255, 255]),
    'jaune': ([20,  100, 100], [35,  255, 255]),
    'rose':  ([140, 50,  50],  [170, 255, 255]),
}

def track(image, color='rose'):
    lower, upper = COLOR_RANGES[color]
    blur  = cv2.GaussianBlur(image, (5, 5), 0)
    hsv   = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask  = cv2.erode(mask,  None, iterations=2)
    mask  = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    ctr = (-1, -1)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) > 500:
            M  = cv2.moments(biggest)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            ctr = (cx, cy)
            cv2.circle(image, ctr, 10, (0, 0, 125), -1)
            cv2.drawContours(image, [biggest], -1, (0, 255, 0), 2)
            # Affiche l'aire du contour
            aire = cv2.contourArea(biggest)
            cv2.putText(image, f"Aire: {int(aire)}px", (cx + 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 125), 2)

    cv2.imshow(WINDOW_NAME, image)
    if cv2.waitKey(1) & 0xFF == 27:
        return None
    return ctr

if __name__ == '__main__':
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    VIDEO_PATH = os.path.join(BASE_DIR, 'ball4.mp4')

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir ball4.mp4")
        exit()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fin de vidéo")
            break
        result = track(frame, color='rose')
        if result is None:
            break

    cap.release()
    cv2.destroyAllWindows()