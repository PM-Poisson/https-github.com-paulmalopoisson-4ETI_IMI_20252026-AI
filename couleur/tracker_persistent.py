import cv2
import numpy as np
import os

WINDOW_NAME = 'BallTracker - Persistant'

COLOR_RANGES = {
    'vert':  ([40,  70,  70],  [80,  255, 255]),
    'bleu':  ([100, 100, 50],  [130, 255, 255]),
    'jaune': ([20,  100, 100], [35,  255, 255]),
    'rose':  ([140, 50,  50],  [170, 255, 255]),
}

last_pos = None

def track(image, color='rose'):
    global last_pos

    lower, upper = COLOR_RANGES[color]
    blur  = cv2.GaussianBlur(image, (5, 5), 0)
    hsv   = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask  = cv2.erode(mask,  None, iterations=2)
    mask  = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Première détection : on prend le plus grand
        if last_pos is None:
            chosen = max(contours, key=cv2.contourArea)
        else:
            def dist_to_last(c):
                M = cv2.moments(c)
                if M['m00'] == 0:
                    return float('inf')
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx - last_pos[0])**2 + (cy - last_pos[1])**2
            chosen = min(contours, key=dist_to_last)

        if cv2.contourArea(chosen) > 500:
            M  = cv2.moments(chosen)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            last_pos = (cx, cy)

            cv2.circle(image, last_pos, 10, (0, 0, 125), -1)
            cv2.drawContours(image, [chosen], -1, (0, 255, 0), 2)
            cv2.putText(image, f"Cible : {last_pos}", (cx + 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 125), 2)

    cv2.imshow(WINDOW_NAME, image)
    if cv2.waitKey(1) & 0xFF == 27:
        return None
    return last_pos if last_pos else (-1, -1)

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