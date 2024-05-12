import time

from numba import jit
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

    def append(self, frame, lane_mask=None, pos=None):
        if pos is not None:
            if len(self.rotation_list_median) < pos-1:
                self.rotation_list_median.extend([np.nan] * (pos - len(self.rotation_list_median)))
                self.rotation_list_mean.extend([np.nan] * (pos - len(self.rotation_list_mean)))
        return self.get_rotation(frame, lane_mask)

    def calculate_line_angles(self, linesP):
        line_angles = []
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            line_angles.append(angle)
        return line_angles

    def get_rotation(self, frame, lane_mask=None):
        frame_masked = self.cf.get_lane_mask(frame) if lane_mask is None else lane_mask

        # Resize frame_masked if necessary
        if self.scale_factor != 1.0:
            frame_masked = cv2.resize(frame_masked, None, fx=self.scale_factor, fy=self.scale_factor)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(frame_masked, cv2.MORPH_GRADIENT, kernel)

        linesP = cv2.HoughLinesP(edges, 1, self.houghPrecisionDegree, 100, None,
                                 minLineLength=frame.shape[0] / 3, maxLineGap=frame.shape[0] / 25)

        self.lineImage = frame if self.scale_factor == 1.0 else cv2.resize(frame, None, fx=self.scale_factor,
                                                                           fy=self.scale_factor)

        line_angles = []
        if linesP is not None:
            line_angles = self.calculate_line_angles(linesP)

            for line in linesP:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.lineImage, (x1, y1), (x2, y2), (0, 255, 0), 2)

        median_angle = np.median(line_angles) if line_angles else 0
        self.rotation_list_median.append(median_angle)
        return self.rotation_list_median[-1]

    # def get_rotation(self, frame, lane_mask=None):
    #     if lane_mask is None:
    #         frame_masked = self.cf.get_lane_mask(frame)
    #     else:
    #         frame_masked = lane_mask
    #
    #     # Resize frame_masked
    #     frame_masked = cv2.resize(frame_masked, None, fx=self.scale_factor, fy=self.scale_factor)
    #
    #     kernel = np.ones((3, 3), np.uint8)
    #
    #     # Apply morphology operation
    #     edges = cv2.morphologyEx(frame_masked, cv2.MORPH_GRADIENT, kernel)
    #
    #     # Detect lines
    #     linesP = cv2.HoughLinesP(edges, 1, self.houghPrecisionDegree, 100, None,
    #                              minLineLength=frame.shape[0] / 3, maxLineGap=frame.shape[0] / 25)
    #
    #     # Resize frame
    #     self.lineImage = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor)
    #
    #     line_angles = []
    #     if linesP is not None:
    #         for line in linesP:
    #             x1, y1, x2, y2 = line[0]
    #             angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    #             line_angles.append(angle)
    #             # Draw lines on the image
    #             cv2.line(self.lineImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #     # Calculate median and mean of line angles
    #     median_angle = np.median(line_angles) if line_angles else 0
    #     mean_angle = np.mean(line_angles) if line_angles else 0
    #
    #     self.rotation_list_median.append(median_angle)
    #     self.rotation_list_mean.append(mean_angle)
    #
    #     return self.rotation_list_median[-1]

    def get_mean_rotation(self):
        return self.rotation_list_mean

    def get_median_rotation(self):
        return self.rotation_list_median

    def interpolate_nan(self):
        # Convert the list to a NumPy array
        data_array = np.array(self.rotation_list_median)

        # Find indices of NaN values
        nan_indices = np.isnan(data_array)

        # Generate indices for non-NaN values
        indices = np.arange(len(data_array))

        # Interpolate NaN values using linear interpolation
        data_array[nan_indices] = np.interp(indices[nan_indices], indices[~nan_indices], data_array[~nan_indices])

        # Convert the interpolated array back to a list
        interpolated_data = data_array.tolist()

        self.rotation_list_median = interpolated_data
        return interpolated_data

    def get_filtered_list(self):
        fs = 15.0
        cutoff = 2
        nyq = 0.5 * fs
        order = 11

        normal_cutoff = cutoff / nyq

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, self.rotation_list_median)

        return y
