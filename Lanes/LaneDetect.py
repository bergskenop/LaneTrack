import time

import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from ColorFilter import Colorfilter


class LaneDetect:
    def __init__(self, houghPrecisionDegree=None):
        self.rotation_list_median = []
        self.rotation_list_mean = []
        self.cf = Colorfilter()
        self.scale_factor = 0.4
        self.houghPrecisionDegree = houghPrecisionDegree if houghPrecisionDegree is not None else np.pi / 360
        self.lineImage = None
        self.containsRed = []
        self.points = []

    def append(self, frame, lane_mask=None):
        return self.get_rotation(frame, lane_mask)

    def get_rotation(self, frame, lane_mask=None):
        start = time.time()
        if lane_mask is None:
            frame_masked = self.cf.get_lane_mask(frame)
        else:
            frame_masked = lane_mask
        frame_masked = cv2.resize(frame_masked, None, fx=self.scale_factor, fy=self.scale_factor)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(frame_masked, cv2.MORPH_GRADIENT, kernel)
        linesP = cv2.HoughLinesP(edges, 1, self.houghPrecisionDegree, 100, None, minLineLength=frame.shape[0] / 3,
                                 maxLineGap=frame.shape[0] / 25)
        line_angles = []
        self.lineImage = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor)
        if linesP is not None:
            for line in linesP:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                line_angles.append(angle)
                cv2.line(self.lineImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.rotation_list_median.append(np.median(line_angles))
            self.rotation_list_mean.append(np.mean(line_angles))
        print(f"\tdetecting hough lines took: {time.time()-start}s")
        return self.rotation_list_median[-1]

    def get_mean_rotation(self):
        return self.rotation_list_mean

    def get_median_rotation(self):
        return self.rotation_list_median

    def get_filtered_list(self):
        fs = 60.0
        cutoff = 2
        nyq = 0.5 * fs
        order = 11

        normal_cutoff = cutoff / nyq

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, self.rotation_list_median)

        return y
