from ultralytics import YOLO
from rich.traceback import install
install()
import cv2
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
model=YOLO(str(ROOT / "runs/detect/train/weights/best.pt"))
def cv_rectangle(path):
    """根据给定的图片路径，使用yolo模型进行目标检测，并在检测到的目标上绘制矩形框"""
    results=model.predict(source=str(ROOT / path),conf=0.5)

    for i in range(len(results)):
        xyxy=results[i].boxes.xyxy.tolist()
        img=cv2.imread(str(ROOT / path))
        for j in range(len(xyxy)):
            x1,y1,x2,y2=xyxy[j]
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,200,0),3)
            cv2.putText(img,f"{results[i].names[int(results[i].boxes.cls[j])]}:{results[i].boxes.conf[j]:.2f}",(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,0),2)
        cv2.imwrite(f"output_{time.time()}.jpg",img)

cv_rectangle("exercise1/question/2.3_label_practice_01.jpg")
