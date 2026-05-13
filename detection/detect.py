import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--classes", nargs="+", type=str, default=None)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-json", type=str, default=None, metavar="FILE")
    parser.add_argument("--save-crops", type=str, default=None, metavar="DIR")
    return parser.parse_args()

def resolve_class_ids(model, names):
    if names is None:
        return None
    name_to_id = {n.lower(): i for i, n in model.names.items()}
    ids = [name_to_id[n.lower()] for n in names if n.lower() in name_to_id]
    if not ids:
        print("No valid classes found.")
        sys.exit(1)
    return ids

def annotate(results, class_ids):
    if class_ids is None:
        return results[0].plot()
    result = results[0]
    boxes = result.boxes
    keep = [i for i, c in enumerate(boxes.cls.tolist()) if int(c) in class_ids]
    if not keep:
        return result.orig_img.copy()
    result.boxes = boxes[keep]
    out = result.plot()
    result.boxes = boxes
    return out

def extract_detections(result, class_names, class_ids_filter):
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        if class_ids_filter and cls_id not in class_ids_filter:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id": cls_id,
            "class_name": class_names[cls_id],
            "confidence": round(float(box.conf), 4),
            "bbox": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "width": round(x2 - x1, 1),
                "height": round(y2 - y1, 1),
            }
        })
    return detections

def save_crops(frame, result, class_names, class_ids_filter, out_dir, prefix=""):
    crops_by_class = {}
    counts = {}
    for box in result.boxes:
        cls_id = int(box.cls)
        if class_ids_filter and cls_id not in class_ids_filter:
            continue
        cls_name = class_names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        cls_dir = out_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        idx = counts.get(cls_name, 0)
        cv2.imwrite(str(cls_dir / f"{prefix}{idx:04d}.jpg"), crop)
        counts[cls_name] = idx + 1
        crops_by_class.setdefault(cls_name, []).append(crop)
    return crops_by_class


def build_mosaic(crops, tile_size=(128, 128)):
    n = len(crops)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    tw, th = tile_size
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        canvas[r*th:(r+1)*th, c*tw:(c+1)*tw] = cv2.resize(crop, (tw, th))
    return canvas

def save_mosaics(crops_by_class, out_dir):
    mosaic_dir = out_dir / "mosaics"
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    for cls_name, crops in crops_by_class.items():
        path = mosaic_dir / f"{cls_name}.jpg"
        cv2.imwrite(str(path), build_mosaic(crops))
        print(f"Mosaic: {path} ({len(crops)} tiles)")

def run_image(model, source, class_ids, conf, save_json, crops_dir):
    frame = cv2.imread(source)
    if frame is None:
        print(f"Cannot read image: {source}")
        sys.exit(1)
    results = model(frame, conf=conf)
    cv2.imshow(source, annotate(results, class_ids))

    if save_json:
        h, w = frame.shape[:2]
        data = {
            "source": source,
            "type": "image",
            "frame_size": {"width": w, "height": h},
            "detections": extract_detections(results[0], model.names, class_ids)
        }
        with open(save_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {save_json}")

    if crops_dir:
        out_dir = Path(crops_dir)
        crops_by_class = save_crops(frame, results[0], model.names, class_ids, out_dir)
        save_mosaics(crops_by_class, out_dir)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(model, class_ids, conf, save_json, crops_dir):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)

    all_frames = []
    all_crops_by_class = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf)
        cv2.imshow("YOLOv8", annotate(results, class_ids))

        if save_json:
            h, w = frame.shape[:2]
            all_frames.append({
                "frame": frame_idx,
                "timestamp": round(time.time(), 3),
                "frame_size": {"width": w, "height": h},
                "detections": extract_detections(results[0], model.names, class_ids)
            })

        if crops_dir:
            out_dir = Path(crops_dir)
            crops = save_crops(frame, results[0], model.names, class_ids, out_dir, prefix=f"f{frame_idx:04d}_")
            for cls_name, imgs in crops.items():
                all_crops_by_class.setdefault(cls_name, []).extend(imgs)

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

    if save_json:
        data = {"source": "webcam", "type": "stream", "frames": all_frames}
        with open(save_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {save_json} ({len(all_frames)} frames)")

        if crops_dir and all_crops_by_class:
            save_mosaics(all_crops_by_class, Path(crops_dir))


def main():
    args = parse_args()
    model = YOLO(args.model)
    class_ids = resolve_class_ids(model, args.classes)
    if args.source:
        run_image(model, args.source, class_ids, args.conf, args.save_json, args.save_crops)
    else:
        run_webcam(model, class_ids, args.conf, args.save_json, args.save_crops)


if __name__ == "__main__":
    main()