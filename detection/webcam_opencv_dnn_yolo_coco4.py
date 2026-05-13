#!/usr/bin/env python3

"""webcam_opencv_dnn_yolo_coco.py: détection YOLO via OpenCV DNN,
   sur webcam ou image statique, avec filtre optionnel de classes,
   export JSON, sauvegarde des bounding boxes et mosaïques par classe."""

__author__  = "Fabrice Jumel"
__license__ = "CC0"
__version__ = "0.4"

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse                    # lecture des arguments en ligne de commande
import sys                         # sys.exit() pour quitter proprement
import json                        # export des détections au format JSON
import os                          # création de dossiers pour les crops
from datetime import datetime      # horodatage des détections

import cv2
import numpy as np

# ── Seuils de détection ───────────────────────────────────────────────────────
CONF_THRESHOLD = 0.5   # confiance minimale pour garder une détection
NMS_THRESHOLD  = 0.4   # seuil NMS pour supprimer les boîtes redondantes

# ── Taille des vignettes dans la mosaïque ────────────────────────────────────
MOSAIC_TILE_W = 100    # largeur  de chaque vignette (pixels)
MOSAIC_TILE_H = 100    # hauteur  de chaque vignette (pixels)
MOSAIC_COLS   = 5      # nombre de colonnes dans la mosaïque


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
        nargs="+",
        default=None,
        help="Liste des classes à afficher, ex : --classes person car dog"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="detections.json",
        help="Chemin du fichier JSON de sortie (défaut : detections.json)"
    )
    parser.add_argument(
        "--output-crops",
        type=str,
        default="crops",
        help="Dossier de sortie pour les bounding boxes découpées (défaut : crops/)"
    )

    return parser.parse_args()


# ── Ouverture de la source vidéo/image ────────────────────────────────────────
def open_source(source):
    """
    Ouvre la source selon l'argument fourni.
    - Retourne (cap, None)   pour une webcam (lecture continue).
    - Retourne (None, frame) pour une image statique (lecture unique).
    """
    if source is None or source.isdigit():
        cam_idx = 0 if source is None else int(source)
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        if not cap.isOpened():
            sys.exit(f"[ERREUR] Impossible d'ouvrir la webcam {cam_idx}")
        return cap, None

    frame = cv2.imread(source)
    if frame is None:
        sys.exit(f"[ERREUR] Impossible de lire l'image : {source}")
    return None, frame


# ── Sauvegarde du JSON ────────────────────────────────────────────────────────
def save_json(data, path):
    """Écrit le dictionnaire `data` dans un fichier JSON indenté."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Détections sauvegardées dans : {path}")


# ── Sauvegarde d'un crop (bounding box découpée) ─────────────────────────────
def save_crop(frame, x, y, w, h, label, frame_idx, crop_idx, output_dir):
    """
    Découpe la région [y:y+h, x:x+w] dans la frame et la sauvegarde en JPEG.
    Les crops sont organisés dans des sous-dossiers par classe :
      output_dir/
        dog/
          dog_f0_0.jpg   (frame 0, détection 0)
          dog_f0_1.jpg
        person/
          person_f1_0.jpg
    Retourne le chemin du fichier créé, ou None si le crop est invalide.
    """
    # Clamp : s'assure que la boîte reste dans les limites de l'image
    img_h, img_w = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    # On ignore les boîtes vides ou dégénérées
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]   # découpe de la région d'intérêt

    # Création du sous-dossier par classe si nécessaire
    class_dir = os.path.join(output_dir, label)
    os.makedirs(class_dir, exist_ok=True)

    filename = f"{label}_f{frame_idx}_{crop_idx}.jpg"
    filepath = os.path.join(class_dir, filename)
    cv2.imwrite(filepath, crop)

    return filepath


# ── Génération de la mosaïque pour une classe ─────────────────────────────────
def save_mosaic(crops_by_class, output_dir):
    """
    Pour chaque classe, assemble toutes ses vignettes dans une image mosaïque
    et la sauvegarde sous : output_dir/<classe>/<classe>_mosaic.jpg

    Disposition : MOSAIC_COLS colonnes, autant de lignes que nécessaire.
    Les vignettes sont toutes redimensionnées à MOSAIC_TILE_W × MOSAIC_TILE_H.
    Les cases vides (dernière ligne incomplète) sont remplies en noir.
    """
    for label, crop_list in crops_by_class.items():
        if not crop_list:
            continue

        n      = len(crop_list)
        cols   = min(n, MOSAIC_COLS)                   # colonnes effectives
        rows   = (n + cols - 1) // cols                # lignes nécessaires

        # Création d'un canvas noir de la bonne taille
        mosaic = np.zeros((rows * MOSAIC_TILE_H,
                           cols * MOSAIC_TILE_W, 3), dtype=np.uint8)

        for idx, crop_img in enumerate(crop_list):
            row = idx // cols
            col = idx  % cols

            # Redimensionnement de la vignette
            tile = cv2.resize(crop_img, (MOSAIC_TILE_W, MOSAIC_TILE_H))

            # Placement dans le canvas
            y_off = row * MOSAIC_TILE_H
            x_off = col * MOSAIC_TILE_W
            mosaic[y_off:y_off + MOSAIC_TILE_H,
                   x_off:x_off + MOSAIC_TILE_W] = tile

        # Sauvegarde de la mosaïque dans le dossier de la classe
        mosaic_path = os.path.join(output_dir, label, f"{label}_mosaic.jpg")
        cv2.imwrite(mosaic_path, mosaic)
        print(f"[INFO] Mosaïque ({n} vignettes) : {mosaic_path}")


# ── Programme principal ───────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Chargement du réseau YOLO (poids + architecture)
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

    # Chargement des noms de classes COCO (80 classes)
    with open("coco.names", "r") as f:
        all_classes = f.read().strip().split("\n")

    # Construction du filtre de classes (optionnel)
    filter_ids = None
    if args.classes is not None:
        filter_ids = {all_classes.index(c) for c in args.classes if c in all_classes}
        unknown = [c for c in args.classes if c not in all_classes]
        if unknown:
            print(f"[ATTENTION] Classes inconnues ignorées : {unknown}")

    # Ouverture de la source
    cap, static_frame = open_source(args.source)

    # Création du dossier de crops racine
    os.makedirs(args.output_crops, exist_ok=True)

    # ── Structure JSON racine ─────────────────────────────────────────────────
    json_output = {
        "source": args.source if args.source is not None else "webcam",
        "date":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frames": []
    }

    # ── Accumulateur de crops par classe (pour les mosaïques) ─────────────────
    # Dictionnaire  { "dog": [img1, img2, …], "person": […], … }
    # On stocke les images en mémoire et on génère les mosaïques à la fin.
    crops_by_class = {}

    frame_index = 0

    # ── Boucle principale ─────────────────────────────────────────────────────
    while True:

        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fin du flux webcam.")
                break
        else:
            frame = static_frame

        height, width = frame.shape[:2]

        # Blob → inférence
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores     = detection[5:]
                class_id   = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < CONF_THRESHOLD:
                    continue
                if filter_ids is not None and class_id not in filter_ids:
                    continue

                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w  = int(detection[2] * width)
                h  = int(detection[3] * height)
                x  = cx - w // 2
                y  = cy - h // 2

                class_ids.append(class_id)
                confidences.append(confidence)
                boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                   CONF_THRESHOLD, NMS_THRESHOLD)

        # ── Traitement de chaque détection retenue ────────────────────────────
        frame_detections = []
        for crop_idx, i in enumerate(indices):
            x, y, w, h = boxes[i]
            label = all_classes[class_ids[i]]

            # Dessin sur l'image affichée
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # ── Sauvegarde du crop JPEG ───────────────────────────────────────
            # On sauvegarde la région découpée dans crops/<classe>/<nom>.jpg
            crop_path = save_crop(frame, x, y, w, h, label,
                                  frame_index, crop_idx, args.output_crops)

            # ── Accumulation pour la mosaïque ─────────────────────────────────
            # On garde une copie de l'image du crop en mémoire pour assembler
            # la mosaïque finale par classe à la toute fin du programme.
            if crop_path is not None:
                img_h, img_w = frame.shape[:2]
                x1 = max(0, x);  y1 = max(0, y)
                x2 = min(img_w, x + w);  y2 = min(img_h, y + h)
                crop_img = frame[y1:y2, x1:x2].copy()  # .copy() pour éviter les refs
                crops_by_class.setdefault(label, []).append(crop_img)

            # JSON
            frame_detections.append({
                "class":      label,
                "confidence": round(confidences[i], 4),
                "bbox":       {"x": x, "y": y, "w": w, "h": h},
                "crop_file":  crop_path   # chemin du JPEG sauvegardé
            })

        json_output["frames"].append({
            "frame_index": frame_index,
            "detections":  frame_detections
        })
        frame_index += 1

        cv2.imshow("Object Detection", frame)

        if cap is not None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            cv2.waitKey(0)
            break

    # ── Sauvegarde finale ─────────────────────────────────────────────────────
    save_json(json_output, args.output_json)   # fichier JSON global

    # Génération des mosaïques : une par classe détectée
    save_mosaic(crops_by_class, args.output_crops)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
