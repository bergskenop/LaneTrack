import time

from matplotlib import pyplot as plt
from ultralytics import YOLO


import cv2
import numpy as np

import Lanes
import Lanes.LaneDetect
import Swimmer
from helper import *
import Frame

selected_contour_index = -1
hovered_contour_index = -1
stable_contours = None

def mouse_callback(event, x, y, flags, param):
    global selected_contour_index, hovered_contour_index
    contours = param['contours']

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the cursor is inside any of the contours
        for i, contour in enumerate(contours):
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                selected_contour_index = i
                break

    if event == cv2.EVENT_MOUSEMOVE:
        # Check if the cursor is inside any of the contours
        for i, contour in enumerate(contours):
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                hovered_contour_index = i
                break
        else:
            hovered_contour_index = -1

class Pool():
    def __init__(self, type= None):
        self.ready = False
        self.remarkable_frames = []
        self.lane_angles = []
        self.Lanes = []
        self.type = type if type is not None else None
        self.selected_lane = None
        self.LD = Lanes.LaneDetect.LaneDetect(houghPrecisionDegree=np.pi / 720)
        self.cf = ColorFilter.Colorfilter()
        self.swimmer = None
        self.model = YOLO('weights/yolov8n_swimmer_v7.pt')
        self.frame_instance = None

    def setup(self):
        self.ready = True

    def process(self, data_path):
        data_type = check_path_type(data_path)
        if data_type == "video":
            for idx, frame in enumerate(read_video(data_path)):
                if self.selected_lane is None:
                    self.select_lane(frame)
                    self.swimmer = Swimmer.Swimmer()
                elif self.swimmer.id is None:
                    start = time.time()
                    self.frame_instance = Frame.Frame(frame, self.model, self.cf, self.LD)
                    self.swimmer.id = self.frame_instance.swimmer_id
                    print(f'Tracking swimmer with id: {self.swimmer.id}')
                    if self.frame_instance.remarkable:
                        self.remarkable_frames.append((idx, self.frame_instance))
                    yield self.frame_instance.filled_image
                    print(f'\rframe {idx} processing time: {round((time.time() - start) * 100, 2)}ms')
                else:
                    start = time.time()
                    self.frame_instance.add_frame(frame, self.model)
                    if self.frame_instance.remarkable:
                        self.remarkable_frames.append((idx, self.frame_instance))
                    annotated_image = put_text_on_image(self.frame_instance.filled_image, self.frame_instance.state_verbose, self.frame_instance.state)
                    display(annotated_image)
                    cv2.waitKey(5)
                    yield annotated_image
                    print(f'\rframe {idx} processing time: {round((time.time() - start) * 100, 2)}ms')
        elif data_type == 'image':
            img = cv2.imread(data_path)
        elif data_type == 'directory':
            for filename in os.listdir(data_path):
                image_path = os.path.join(data_path, filename)
                img = cv2.imread(image_path)

    def select_lane(self, frame):
        """
            1) Create a mask based on the colors of the swimming lanes (CHECK)
            2) Approximate the swimming lanes with lines
            3) Approximate the end of the swimming pool with a line
        """
        frame_cpy = frame.copy()
        mask = self.cf.get_swimming_pool(frame_cpy)

        contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = validate_contours(frame, contours)
        print(f'{len(valid_contours)}: valid out of {len(contours)} contours')
        lane_selected = False
        if selected_contour_index == -1:
            cv2.namedWindow('Contour Selection', cv2.WINDOW_KEEPRATIO)
            cv2.setMouseCallback('Contour Selection', mouse_callback, {'contours': valid_contours})

        while not lane_selected and selected_contour_index == -1:
            mask = np.zeros_like(frame_cpy)
            if selected_contour_index != -1 and len(valid_contours) >= selected_contour_index:
                lane_selected = True
                cv2.drawContours(mask, valid_contours, selected_contour_index, (0, 255, 255), thickness=cv2.FILLED)
            if hovered_contour_index != -1 and len(valid_contours) >= hovered_contour_index:
                cv2.drawContours(mask, valid_contours, hovered_contour_index, (0, 255, 0), thickness=cv2.FILLED)

            alpha = 0.25
            filled_image = cv2.addWeighted(frame_cpy, 1 - alpha, mask, alpha, 0)
            cv2.imshow('Contour Selection', filled_image)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cv2.destroyWindow('Contour Selection')
        if valid_contours is not None:
            self.Lanes = valid_contours
            self.selected_lane = Lanes.Lane(frame, valid_contours[selected_contour_index])

        print(f"Swimming pool has been initiated with {len(self.Lanes)} visible lanes.")

    def test_lanemaskdetection(self, data_path):
        data_type = check_path_type(data_path)
        if data_type == "video":
            print(f'Saving lane masks to video file')
            for idx, frame in enumerate(read_video(data_path)):
                lane_mask = self.cf.get_lane_mask(frame, visual=True)
                display(lane_mask)
                cv2.waitKey(5)


