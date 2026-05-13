import cv2
import numpy as np
import os

WINDOW_NAME = 'BallTracker - Apprentissage'

learned_lower = None
learned_upper = None

def learn_color_from_center(image, margin=20):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    roi = image[cy-margin:cy+margin, cx-margin:cx+margin]

    # Dessine le carré de sélection
    cv2.rectangle(image, (cx-margin, cy-margin),
                         (cx+margin, cy+margin), (0, 255, 0), 2)
    cv2.putText(image, "Place la balle ici et appuie sur ESPACE",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels  = hsv_roi.reshape(-1, 3)
    lower   = np.clip(pixels.min(axis=0) - [10, 40, 40], 0, 255)
    upper   = np.clip(pixels.max(axis=0) + [10, 40, 40], 0, 255)
    return lower.astype(np.uint8), upper.astype(np.uint8)

def track(image, lower, upper):
    blur  = cv2.GaussianBlur(image, (5, 5), 0)
    hsv   = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, lower, upper)
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

    cv2.imshow(WINDOW_NAME, image)
    return ctr

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam")
        exit()

    print("=== MODE APPRENTISSAGE ===")
    print("Place la balle au centre du carré vert et appuie sur ESPACE")
    print("Appuie sur ECHAP pour quitter")

    lower, upper = None, None
    learning = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if learning:
            # Affiche le carré et calcule en temps réel
            lower, upper = learn_color_from_center(frame)
            cv2.imshow(WINDOW_NAME, frame)
        else:
            # Mode tracking
            cv2.putText(frame, "MODE TRACKING", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            track(frame, lower, upper)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Echap
            break
        elif key == 32 and learning:  # Espace
            print(f"Couleur apprise ! Lower={lower}, Upper={upper}")
            learning = False

    cap.release()
    cv2.destroyAllWindows()