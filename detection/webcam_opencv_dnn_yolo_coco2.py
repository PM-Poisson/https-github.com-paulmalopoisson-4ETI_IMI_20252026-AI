#!/usr/bin/env python3

"""webcam_opencv_dnn_yolo_coco.py: détection YOLO via OpenCV DNN,
   sur webcam ou image statique, avec filtre optionnel de classes."""

__author__  = "Fabrice Jumel"
__license__ = "CC0"
__version__ = "0.2"

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse          # lecture des arguments en ligne de commande
import sys               # sys.exit() pour quitter proprement en cas d'erreur

import cv2
import numpy as np

# ── Seuils de détection ───────────────────────────────────────────────────────
CONF_THRESHOLD = 0.5   # confiance minimale pour garder une détection
NMS_THRESHOLD  = 0.4   # seuil NMS pour supprimer les boîtes redondantes


# ── Parsing des arguments CLI ─────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Détection YOLO (webcam ou image)")

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Index webcam (0, 1…) ou chemin vers une image. "
             "Par défaut : webcam 0."
    )
    parser.add_argument(
        "--classes",
        nargs="+",       # accepte 1 ou plusieurs valeurs : --classes person car
        default=None,    # None = pas de filtre, toutes les classes sont affichées
        help="Liste des classes à afficher, ex : --classes person car dog"
    )

    return parser.parse_args()


# ── Ouverture de la source vidéo/image ────────────────────────────────────────
def open_source(source):
    """
    Ouvre la source selon l'argument fourni.
    - Retourne (cap, None)   pour une webcam (lecture continue).
    - Retourne (None, frame) pour une image statique (lecture unique).
    """
    # Pas de source précisée, ou un chiffre → webcam
    if source is None or source.isdigit():
        cam_idx = 0 if source is None else int(source)
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        if not cap.isOpened():
            sys.exit(f"[ERREUR] Impossible d'ouvrir la webcam {cam_idx}")
        return cap, None

    # Sinon on tente de lire une image depuis le chemin fourni
    frame = cv2.imread(source)
    if frame is None:
        sys.exit(f"[ERREUR] Impossible de lire l'image : {source}")
    return None, frame


# ── Programme principal ───────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Chargement du réseau YOLO (poids + architecture)
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

    # Chargement des noms de classes COCO (80 classes)
    with open("coco.names", "r") as f:
        all_classes = f.read().strip().split("\n")

    # Construction du filtre de classes à partir des arguments CLI
    # ex : --classes person car  →  filter_ids = {0, 2}
    filter_ids = None
    if args.classes is not None:
        filter_ids = {all_classes.index(c) for c in args.classes if c in all_classes}
        unknown = [c for c in args.classes if c not in all_classes]
        if unknown:
            print(f"[ATTENTION] Classes inconnues ignorées : {unknown}")

    # Ouverture de la source (webcam ou image)
    cap, static_frame = open_source(args.source)

    # ── Boucle principale ─────────────────────────────────────────────────────
    while True:

        # Lecture de la frame selon la source
        if cap is not None:
            ret, frame = cap.read()   # webcam : nouvelle frame à chaque itération
            if not ret:
                print("[INFO] Fin du flux webcam.")
                break
        else:
            frame = static_frame      # image : on réutilise la même frame

        height, width = frame.shape[:2]

        # Conversion de la frame en blob (entrée du réseau)
        # - normalisation pixel [0,1]  (÷ 255)
        # - redimensionnement à 416×416 (taille attendue par YOLOv3-tiny)
        # - swapRB=True : OpenCV lit BGR, YOLO attend RGB
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)

        # Inférence : récupération des sorties des couches YOLO finales
        outs = net.forward(net.getUnconnectedOutLayersNames())

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores     = detection[5:]           # scores par classe
                class_id   = int(np.argmax(scores))  # classe la plus probable
                confidence = float(scores[class_id]) # score associé

                # On ignore les détections sous le seuil de confiance
                if confidence < CONF_THRESHOLD:
                    continue

                # On ignore les classes non demandées (si filtre actif)
                if filter_ids is not None and class_id not in filter_ids:
                    continue

                # Conversion des coordonnées relatives → pixels
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w  = int(detection[2] * width)
                h  = int(detection[3] * height)
                x  = cx - w // 2   # coin supérieur gauche
                y  = cy - h // 2

                class_ids.append(class_id)
                confidences.append(confidence)
                boxes.append([x, y, w, h])

        # Suppression des boîtes redondantes (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                   CONF_THRESHOLD, NMS_THRESHOLD)

        # Dessin des boîtes conservées
        for i in indices:
            x, y, w, h = boxes[i]
            label = all_classes[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        cv2.imshow("Object Detection", frame)

        # Condition de sortie
        if cap is not None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break          # webcam : appuyer sur 'q' pour quitter
        else:
            cv2.waitKey(0)     # image  : n'importe quelle touche pour quitter
            break

    # Libération des ressources
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()