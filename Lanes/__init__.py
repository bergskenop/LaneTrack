import time

import cv2
import numpy as np
from helper import *
import ColorFilter

class Lane:
    def __init__(self, frame, contour):
        self.lane_n = None
        self.frame = frame
        self.lane_mask = np.zeros_like(frame)
        self.contour_area = cv2.contourArea(contour)
        cv2.drawContours(self.lane_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        self.red_lines = []
        self.cf = ColorFilter.Colorfilter()
        self.visual_lane = None

    def track_lane(self, frame):
        frame_cpy = frame.copy()
        lane_mask = self.cf.get_lanes(frame_cpy, redMode=False)[1]
        border_mask = self.cf.swimming_pool_box(frame_cpy)[0]
        if border_mask is not None:
            mask = cv2.bitwise_or(lane_mask, border_mask)
            mask = cv2.bitwise_not(mask)
        else:
            mask = cv2.bitwise_not(lane_mask)
        contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = validate_contours(frame, contours, self.contour_area)
        if valid_contours is not None:
            for contour in valid_contours:
                frame_mask_cpy = frame.copy()
                new_mask = np.zeros_like(frame_mask_cpy)
                cv2.drawContours(new_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                alpha = 0.25
                overlap = cv2.bitwise_and(new_mask, self.lane_mask)

                overlap_percentage = cv2.countNonZero(overlap[:, :, 0])/cv2.countNonZero(self.lane_mask[:, :, 0])
                if overlap_percentage > 0.8:
                    self.contour_area = cv2.contourArea(contour)
                    filled_image = cv2.addWeighted(frame, 1 - alpha, new_mask, alpha, 0)
                    self.visual_lane = filled_image
                    self.lane_mask = new_mask
                    break
        else:
            alpha = 0.25
            filled_image = cv2.addWeighted(frame, 1 - alpha, self.lane_mask, alpha, 0)
            self.visual_lane = filled_image


    def get_red_lines(self):
        pass
