from ultralytics import YOLO
import cv2
import gc
import time

async def ClearMemory():
    while True:
        gc.collect()
        time.sleep(1)

def main():
    model = YOLO('yolov8n.pt')
    ClearMemory()
    for results in model.track(source="traffic.mp4", imgsz=320, show=True, stream=True):
        frame = results.orig_img

        cv2.imshow('yolov8', frame)

        if (cv2.waitKey(30)==27):
            break


if __name__ == '__main__':
    main()