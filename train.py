from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("yolov10n.pt")

# Train the model
model.train(data="label studio/data.yaml", epochs=30, imgsz=640)