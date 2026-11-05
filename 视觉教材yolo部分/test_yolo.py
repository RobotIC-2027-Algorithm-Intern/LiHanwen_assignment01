from ultralytics import YOLO
from rich.traceback import install
install()
model =YOLO("runs/detect/train/weights/best.pt")
results = model.predict(source="exercise2/VID_20260506_185211.mp4",conf=0.25,stream=True)
count = 0
for result in results:
    count +=1
    if count %30 ==0:
        result.save(filename=f"exercise2/{count}.jpg")
        