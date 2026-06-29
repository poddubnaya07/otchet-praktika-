import cv2
from src.train import yolo

def detect_people(img_path, conf=0.25):
    img = cv2.imread(img_path)
    results = yolo(img, conf=conf, verbose=False)
    
    people = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                if box.cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    people.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(box.conf[0])
                    })
    return people, len(people)

print("Инференс загружен!")
