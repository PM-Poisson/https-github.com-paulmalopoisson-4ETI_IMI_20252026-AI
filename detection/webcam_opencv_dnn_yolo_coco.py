#!/usr/bin/env python3

"""webcam_opencv_dnn_yolo_coco.py: test of opencv yolo with a webcam"""

__author__      = "Fabrice Jumel"
__license__ = "CC0"
__version__ = "0.1"

import cv2
import numpy as np




def parse_args():
    parser = argparse.ArgumentParser(description="Détection YOLO")
    
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Webcam (None ou 0) ou chemin vers une image"
    )
    parser.add_argument(
        "--classes",
        nargs="+",        # accepte 1 ou plusieurs valeurs : --classes person car
        default=None,     # None = pas de filtre, toutes les classes passent
        help="Classes à détecter ex: --classes person car"
    )
    
    return parser.parse_args()

def open_source(source):
    """
    Retourne soit (cap, None) pour une webcam
    soit (None, frame) pour une image statique
    """
    # Pas de source précisée = webcam par défaut
    if source is None or source.isdigit():
        cam_idx = 0 if source is None else int(source)
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        return cap, None
    
    # Sinon on essaie de lire une image
    frame = cv2.imread(source)
    if frame is None:
        sys.exit(f"[ERREUR] Impossible de lire : {source}")
    
    return None, frame
# Load YOLOv3 model with OpenCV
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam (change as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);

args = parse_args()

# Construction du filtre : ["person", "car"] → {0, 2}
filter_ids = None
if args.classes is not None:
    filter_ids = {classes.index(c) for c in args.classes if c in classes}

cap, static_frame = open_source(args.source)
# cap.set(cv2.CAP_PROP_FPS,120)
# example resolution logi hd1080 : 2304x1536 2304x1296 1920x1080 1600x896 1280x720 960x720 1024x576 800x600 864x480 800x448 640x480 640x360 432x240 352x288 320x240 320x180 176x144 160x120 160x90
# example resolution logi 720 : 1280x960 1280x720 1184x656 960x720 1024x576 800x600 864x480 800x448 752x416 640x480 640x360 432x240 352x288 320x240 320x176 176x144 160x120 160x90
while True:
    # ── Lecture de la frame ──────────────────────────────────────
    if cap is not None:
        ret, frame = cap.read()   # webcam : on lit en continu
        if not ret:
            break
    else:
        frame = static_frame      # image : on réutilise la même frame

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids, confidences, boxes = [], [], []
    
    for out in outs:
        for detection in out:
            scores     = detection[5:]
            class_id   = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < conf_threshold:
                continue

            # ── Filtre de classes ────────────────────────────────
            if filter_ids is not None and class_id not in filter_ids:
                continue          # on saute les classes non demandées

            cx = int(detection[0] * width)
            cy = int(detection[1] * height)
            w  = int(detection[2] * width)
            h  = int(detection[3] * height)
            x  = cx - w // 2
            y  = cy - h // 2

            class_ids.append(class_id)
            confidences.append(confidence)
            boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    # ── Condition de sortie selon la source ──────────────────────
    if cap is not None:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break             # webcam : on attend 'q'
    else:
        cv2.waitKey(0)        # image  : on attend n'importe quelle touche
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()

