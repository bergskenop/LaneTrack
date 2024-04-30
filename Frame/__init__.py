import time

import cv2
from ultralytics.utils.plotting import Annotator

from helper import *

import ColorFilter
import Swimmer
import Lanes.LaneDetect


class Frame():
    def __init__(self, frame, model, cf=ColorFilter.Colorfilter(), LD=Lanes.LaneDetect.LaneDetect()):
        self.cf = cf
        self.tracker = 'botsort.yaml'
        self.detection_results = model.track(frame, tracker=self.tracker, verbose=False,
                                             persist=True, conf=0.05, iou=0.2, augment=True,
                                             agnostic_nms=True)[0]
        # Allow user to select swimmer to track
        self.swimmer_id = 2
        self.swimmer_positions = self.detection_results.boxes.xyxy
        self.detected_image = self.detection_results.plot()

        # Mask with the same shape as incoming frame
        self.lane_mask = cf.get_lane_mask(frame, visual=False)
        self.lanes = cf.get_lanes(frame)[1]

        # Mask with marking red segments and the line connecting them
        self.red_segment, self.red_segment_line = cf.get_red_segments(frame, pos="eop")

        # Rough estimate of where frame is located in swimming pool
        # self.lane_angle = LD.append(frame, self.lane_mask)
        self.pool_segment = []

        alpha = 0.25
        red_segment_rgb = cv2.cvtColor(self.red_segment, cv2.COLOR_GRAY2RGB)
        self.filled_image = cv2.addWeighted(frame, 1 - alpha, red_segment_rgb, alpha, 0)

        self.state_verbose = ""
        self.state = 0
        self.remarkable = self.define_remarkable()

    def add_frame(self, frame, model):
        self.detection_results = model.track(frame, tracker=self.tracker, verbose=False,
                                             persist=True, conf=0.05, iou=0.2, augment=True,
                                             agnostic_nms=True)[0]
        self.red_segment, self.red_segment_line = self.cf.get_red_segments(frame, pos="eop")
        # Check if detected result intersects with detected red segment line.
        boxes_id = self.detection_results.boxes.id
        swimmer_box = self.detection_results.boxes.xyxy[(boxes_id == self.swimmer_id).nonzero()]

        if swimmer_box.numel() > 0:
            all_boxes = self.detection_results.boxes.xyxy
            model_image = self.detection_results.plot()
            self.detected_image = draw_box_on_image(model_image, swimmer_box, color=(0, 255, 0))
            alpha = 0.25
            red_segment_rgb = cv2.cvtColor(self.red_segment, cv2.COLOR_GRAY2RGB)
            self.filled_image = cv2.addWeighted(self.detected_image, 1 - alpha, red_segment_rgb, alpha, 0)

            if box_intersects_any_line(swimmer_box, self.red_segment_line):
                self.state_verbose = "Swimmer crossed segment line"
                self.state = 1
                print("box intersected with line")
            else:
                self.state_verbose = "Swimmer detected and tracking"
                self.state = 0
        else:
            alpha = 0.25
            red_segment_rgb = cv2.cvtColor(self.red_segment, cv2.COLOR_GRAY2RGB)
            model_image = self.detection_results.plot()
            self.filled_image = cv2.addWeighted(model_image, 1 - alpha, red_segment_rgb, alpha, 0)
            self.state_verbose = "Swimmer lost"
            self.state = 2
            print("Swimmer ID lost, trying to relocate swimmers by analysing lane")
            # implement lane tracking to keep track of selected lane.
        self.define_remarkable()

    def define_remarkable(self):
        """
        A frame will be defined as remarkable if
            - A detected swimmer passes a segment.
        """
        if self.state == 1:
            self.remarkable = True
        else:
            self.remarkable = False