from ultralytics import YOLO
from helper import *

class Swimmer():
    def __init__(self, model=None):
        self.model = YOLO('weights/yolov8n_swimmer_v8.pt')if model is None else model
        self.position = None
        self.id = None
        self.times = []
        self.tracker = 'botsort.yaml'
        self.box_height_list = []
        self.box_width_list = []

    def detect_swimmer(self, frame):
        results = self.model.track(frame, tracker=self.tracker, verbose=False, persist=True, conf=0.25)
        self.position = results[0].boxes.xyxy
        return results[0]

    def append_box(self, detection_box):
        if detection_box is None:
            self.box_height_list.append(np.nan)
            self.box_width_list.append(np.nan)
        else:
            x1, y1, x2, y2 = detection_box[0, 0, :].cpu().int().tolist()
            width = x2 - x1
            height = y2 - y1
            self.box_width_list.append(width)
            self.box_height_list.append(height)


