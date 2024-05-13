import time

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

from helper import *

import ColorFilter
import Swimmer
import Lanes.LaneDetect


class Frame:
    def __init__(self, frame, model, cf=ColorFilter.Colorfilter(), LD=Lanes.LaneDetect.LaneDetect(), lane=None,
                 lookingAt=2, swimmer_id=None):
        self.cf = cf
        self.tracker = 'botsort.yaml'
        self.detection_results = model.track(frame, tracker=self.tracker, verbose=False,
                                             persist=True, conf=0.05, iou=0.2, augment=True,
                                             agnostic_nms=True)[0]
        # Allow user to select swimmer to track
        self.swimmer = Swimmer.Swimmer()
        self.swimmer.id = swimmer_id
        self.swimmer_positions = self.detection_results.boxes.xyxy
        self.detected_image = self.detection_results.plot()

        # Mask with the same shape as incoming frame
        self.lane_mask = cf.get_lane_mask(frame, visual=False)
        self.lanes = cf.get_lanes(frame)[1]
        self.selected_lane = lane

        # Mask with marking red segments and the line connecting them
        self.red_segment, self.red_segment_line = cf.get_red_segments(frame, lookingAt="eop")

        # Rough estimate of where frame is located in swimming pool
        self.LD = LD
        self.lane_angle = LD.append(frame, self.lane_mask)
        self.lookingAt = lookingAt

        alpha = 0.25
        red_segment_rgb = cv2.cvtColor(self.red_segment, cv2.COLOR_GRAY2RGB)
        self.filled_image = cv2.addWeighted(frame, 1 - alpha, red_segment_rgb, alpha, 0)

        self.state_verbose = []
        self.state = 0

        self.remarkable = False

    def add_frame(self, frame, model, lookingAt):
        self.state_verbose = []
        # region TrackLaneAndSwimmer
        # Track the object ID
        self.selected_lane.track_lane(frame)
        self.detection_results = model.track(frame, tracker=self.tracker, verbose=False,
                                             persist=True, conf=0.05, iou=0.2, augment=True,
                                             agnostic_nms=True)
        boxes_id = self.detection_results[0].boxes.id
        try:
            swimmer_box = self.detection_results[0].boxes.xyxy[(boxes_id == self.swimmer.id).nonzero()]
        except Exception as e:
            print("\nAn error occurred:", str(e))
            return False
        # endregion
        # region GetRedSegments
        self.red_segment, self.red_segment_line = self.cf.get_red_segments(frame, lookingAt=lookingAt)
        # endregion
        if swimmer_box.numel() > 0:
            # region CheckSegmentCrossing
            # region visualisation
            # model_image = self.detection_results[0].plot()
            # self.detected_image = draw_box_on_image(model_image, swimmer_box, color=(0, 255, 0))
            # alpha = 0.25
            # red_segment_rgb = cv2.cvtColor(self.red_segment, cv2.COLOR_GRAY2RGB)
            # segment_line_overlay = cv2.addWeighted(self.detected_image, 1 - alpha, red_segment_rgb, alpha, 0)

            alpha = 0.25
            red_rgb = np.zeros((self.red_segment.shape[0], self.red_segment.shape[1], 3), dtype=np.uint8)
            red_rgb[self.selected_lane.lane_mask[:,:,0] == 255] = (255, 0, 255)
            red_rgb[self.red_segment == 255] = (255, 255, 255)
            model_image = self.detection_results[0].plot()
            model_image = draw_box_on_image(model_image, swimmer_box, color=(0, 255, 0), thickness=3)

            self.filled_image = cv2.addWeighted(model_image, 1 - alpha, red_rgb, alpha, 0)

            # self.filled_image = segment_line_overlay
            # endregion
            if box_intersects_any_line(swimmer_box, self.red_segment_line):
                # TODO issue with line crossing FIX
                self.state_verbose.append("Swimmer crossed segment line")
                self.state = 1
            else:
                self.state_verbose.append("Swimmer detected and tracking")
                self.state = 0
            self.swimmer.append_box(swimmer_box)
        # endregion
        # region SwimmerLostNewID

        else:
            """
            If swimmer is lost:
            This part will check if a bounding box with a new ID of class swimmer can be found within the tracked swimming lane
            """
            tracked_lane = self.selected_lane.lane_mask
            cv2.imshow("tracked_lane", tracked_lane)
            for result in self.detection_results:
                boxes = result.boxes.cpu().numpy()
                xyxys = result.boxes.xyxy
                class_ids = boxes.cls
                object_ids = boxes.id
                for xyxy, cls, object_id in zip(xyxys, class_ids, object_ids):
                    frame_mask = np.zeros_like(frame)
                    cv2.rectangle(frame_mask, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                  (255, 255, 255), cv2.FILLED)
                    overlap_with_lane = cv2.bitwise_and(frame_mask[:, :, 0], tracked_lane[:, :, 0], mask=None)
                    rect_cnt = cv2.countNonZero(frame_mask[:, :, 0])
                    overlap_cnt = cv2.countNonZero(overlap_with_lane)
                    if overlap_cnt / rect_cnt > 0.9 and cls == 1:
                        print(f"new ID found, now tracking swimmer ID: {object_id}")
                        self.swimmer.id = object_id
                        if box_intersects_any_line(xyxy.reshape(1, 1, 4), self.red_segment_line):
                            # TODO issue with line crossing FIX
                            self.state_verbose.append("Swimmer crossed segment line")
                            self.state = 1
                        else:
                            self.state_verbose.append("Swimmer detected and tracking")
                            self.state = 0
                        self.swimmer.append_box(xyxy.reshape(1, 1, 4))
                        break
                # If no swimmer is found append None to swimmer box (interpolation)
                self.swimmer.append_box(None)

            alpha = 0.25
            red_rgb = np.zeros((self.red_segment.shape[0], self.red_segment.shape[1], 3), dtype=np.uint8)
            red_rgb[self.red_segment == 255] = (0, 0, 255)
            red_rgb[self.selected_lane.lane_mask[:, :, 0] == 255] = (255, 0, 255)
            model_image = self.detection_results[0].plot()
            self.filled_image = cv2.addWeighted(model_image, 1 - alpha, red_rgb, alpha, 0)
            self.state_verbose.append("Swimmer lost")
            self.state = 2
            print("Swimmer ID lost, trying to relocate swimmers by analysing lane")
        # endregion
        # region VerboseHandling
        if lookingAt == 0:
            self.state_verbose.append("Looking at right side of pool")
        elif lookingAt == 1:
            self.state_verbose.append("Looking at middle of pool")
        else:
            self.state_verbose.append("Looking at left side of pool")
        # endregion
        # region DefineRemarkable
        if self.state == 1:
            self.remarkable = True
        else:
            self.remarkable = False
        # endregion
        return True

    def define_remarkable(self):
        """
        A frame will be defined as remarkable if
            - A detected swimmer passes a segment.
        """

