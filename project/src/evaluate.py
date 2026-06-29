import time
import numpy as np
import cv2
from src.train import yolo, yolo_large, yolo_v5, detr, hog

def count_yolo(img, model, conf=0.25):
    results = model(img, conf=conf, verbose=False)
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            person_mask = boxes.cls == 0
            return int(person_mask.sum().item())
    return 0

def count_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes, _ = hog.detectMultiScale(gray, winStride=(8,8), padding=(4,4), scale=1.05)
    return len(boxes)

def evaluate_models(img_list):
    models = {
        "YOLOv8n": lambda x: count_yolo(x, yolo, 0.25),
        "YOLOv8x": lambda x: count_yolo(x, yolo_large, 0.25),
        "YOLOv5nu": lambda x: count_yolo(x, yolo_v5, 0.25),
        "RT-DETR": lambda x: count_yolo(x, detr, 0.25),
        "HOG+SVM": count_hog
    }
    
    results = {}
    for name, func in models.items():
        counts, times = [], []
        for img_path in img_list:
            img = cv2.imread(img_path)
            start = time.time()
            count = func(img)
            times.append((time.time() - start) * 1000)
            counts.append(count)
        results[name] = {
            "mean_count": float(np.mean(counts)),
            "max_count": int(np.max(counts)),
            "mean_time_ms": float(np.mean(times))
        }
    return results
