from ultralytics import YOLO
from helper import *

class Swimmer():
    def __init__(self, model=None):
        self.model = YOLO('weights/yolov8m_swimmer_v7.pt')if model is None else model
        self.position = None
        self.id = None
        self.times = []
        self.tracker = 'botsort.yaml'
        self.alpha_blend = 0.25
        self.dilate_kernel = np.ones((3, 3), np.uint8)

    def detect_swimmer(self, frame, mask=None,):
        results = self.model.track(frame, tracker=self.tracker, verbose=False, persist=True, conf=0.25)
        self.position = results[0].boxes.xyxy
        return results[0]
