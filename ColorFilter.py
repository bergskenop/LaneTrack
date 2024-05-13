import cv2
import numpy as np
from scipy.spatial import KDTree

from helper import *


class Colorfilter:
    def __init__(self, color_ranges=None):
        self.color_ranges = color_ranges or {
            'yellow': ([15, 70, 80], [55, 255, 255]),
            'blue': ([100, 100, 90], [130, 180, 255]),
            'red': ([150, 100, 80], [180, 255, 235]),
            'red2': ([0, 100, 80], [10, 255, 235]),
            'green': ([70, 120, 30], [90, 255, 100])
        }
        self.mask = []
        self.visual_mask = []

    def get_lane_mask(self, frame, visual=False, redMode=False):
        # Convert frame to HSV color space
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply color filtering based on redMode
        if redMode:
            result_mask = cv2.inRange(hsv_img, np.array(self.color_ranges['red'][0]),
                                      np.array(self.color_ranges['red'][1]))
        else:
            # Combine masks for each color range
            masks = [cv2.inRange(hsv_img, np.array(min_val), np.array(max_val))
                     for (min_val, max_val) in self.color_ranges.values()]
            result_mask = masks[0]
            for mask in masks[1:]:
                result_mask = cv2.bitwise_xor(result_mask, mask)

        # Median blur to smooth the mask
        mask = cv2.medianBlur(result_mask, 9)

        # # Morphological dilation to enhance edges
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # mask = cv2.dilate(mask, kernel, iterations=4)

        mask = self.remove_small_components(mask)

        # Apply the mask to the frame
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_contour_area = frame.shape[0] * frame.shape[1] / 25
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) <= max_contour_area]

        # Sort contours based on width
        sorted_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[2], reverse=True)

        # Draw rectangles around filtered contours
        filtered_contours = []
        if sorted_contours:
            max_width = cv2.boundingRect(sorted_contours[0])[2] / 2
            for contour in sorted_contours:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                width, height = cv2.boundingRect(contour)[2:]

                if max_width * 0.80 <= width:
                    filtered_contours.append(box)
                    # cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        # Create a mask with the filtered contours
        filtered_mask = np.zeros_like(mask)
        cv2.drawContours(filtered_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

        return res if visual else filtered_mask

    def get_swimming_pool(self, frame):
        lane_mask = self.get_lanes(frame, redMode=False)[1]
        border_mask = self.swimming_pool_box(frame)[0]

        if border_mask is None:
            return cv2.bitwise_not(lane_mask)
        else:
            mask = cv2.bitwise_or(lane_mask, border_mask)
            return cv2.bitwise_not(mask)

    def get_lanes(self, frame, redMode=False):
        frame_mask = self.get_lane_mask(frame, redMode=redMode)

        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.Canny(frame_mask, threshold1=100, threshold2=255)
        edges = cv2.dilate(edges, kernel)

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 90, 100, None, minLineLength=frame.shape[0] / 20,
                                 maxLineGap=frame.shape[0] / 15)

        lanes_line_list = []
        frame_lanes_mask = np.zeros_like(frame)

        frame_width = frame_lanes_mask.shape[1]
        if linesP is not None:
            for line in linesP:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    angle_rad = np.radians(angle)
                    m = np.tan(angle_rad)
                    b = y1 - m * x1
                    left_point = (0, int(b))
                    right_point = (frame_width - 1, int(m * (frame_width - 1) + b))
                    lanes_line_list.append([left_point[0], left_point[1], right_point[0], right_point[1]])
                    cv2.line(frame_lanes_mask, left_point, right_point, (255, 255, 255), 2)

        return lanes_line_list, frame_lanes_mask

    def swimming_pool_box(self, frame):
        # frame = self.kmeans_image if self.kmeans_image is not None else self.apply_kmeans(frame)

        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_img,
                                  np.array(self.color_ranges['yellow'][0]),
                                  np.array(self.color_ranges['yellow'][1]))

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.Canny(yellow_mask, threshold1=100, threshold2=255)
        edges = cv2.dilate(edges, kernel)

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, minLineLength=frame.shape[0] / 2,
                                 maxLineGap=frame.shape[0] / 20)
        frame_shape = np.zeros_like(frame)

        frame_height = frame_shape.shape[1]

        if linesP is not None:
            for line in linesP:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angle_rad = np.radians(angle)
                m = np.tan(angle_rad)
                b = y1 - m * x1
                left_point = (0, int(b))
                right_point = (frame_height - 1, int(m * (frame_height - 1) + b))
                cv2.line(frame_shape, left_point, right_point, (255, 255, 255), 2)
        else:
            print('No end of pool found')
            return None, None

        return frame_shape, ((x1, y1), (x2, y2))

    def remove_small_components(self, mask):
        """
        Function remove_small_components()
        1) Detects all components within binary image
        2) Sorts components based on components size
        3) Removes all components smaller than 10% of largest component
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        component_sizes = stats[:, -1]
        sorted_indices = np.argsort(component_sizes)[::-1]

        filtered_mask = np.zeros_like(mask)
        if len(sorted_indices) > 1:
            largest_component_area = component_sizes[sorted_indices[1]]
        else:
            largest_component_area = component_sizes[sorted_indices[0]]
        area_threshold = largest_component_area * 0.05

        for index in sorted_indices[1:]:
            if component_sizes[index] >= area_threshold:
                filtered_mask[labels == index] = 255

        return filtered_mask

    def get_red_segments(self, frame, lookingAt=0):
        # TODO Merge segments that are close by one another (occurs at end of lane with flags overlapping)
        # Merging should happen when lokking at right or left side of pool
        frame_segment_mask = self.get_lane_mask(frame, redMode=True)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if lookingAt == 1: # middle of pool is more prone interference from overhanging segments
            frame_segment_mask = cv2.dilate(frame_segment_mask, kernel, iterations=1)
        else:
            frame_segment_mask = cv2.dilate(frame_segment_mask, kernel, iterations=10)

        contours, _ = cv2.findContours(frame_segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        line_list = []

        frame_mask = np.zeros_like(frame_segment_mask)

        if len(contours) > 1:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour_area = cv2.contourArea(sorted_contours[0])
            contours = [cnt for cnt in sorted_contours if cv2.contourArea(cnt) >= 0.6 * largest_contour_area]

            if contours:  # Ensure there are contours after filtering
                if lookingAt == 1: # Handle middle of pool
                    for contour in contours:

                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            centroids.append((cx, cy))

                    # Construct KD-tree for centroids
                    if centroids:
                        kdtree = KDTree(centroids)

                        # Find nearest neighbor for each centroid
                        for i, centroid in enumerate(centroids):
                            closest_index = kdtree.query(centroid, k=2)[1][1]  # Index of the closest neighbor

                            if closest_index < len(centroids):  # Check if closest_index is within valid range
                                closest_centroid = centroids[closest_index]
                                distance = np.linalg.norm(np.array(centroid) - np.array(closest_centroid))
                                if distance < frame.shape[0]/2:
                                    cv2.line(frame_mask, centroid, closest_centroid, (255, 0, 0), 5)
                                    line_list.append((centroid, closest_centroid))
                elif lookingAt == 0:
                    leftmost_points = []
                    rightmost_points = []
                    for contour in contours:
                        if len(contour) >= 5:
                            leftmost_point = tuple(contour[contour[:, :, 0].argmin()][0])
                            rightmost_point = tuple(contour[contour[:, :, 0].argmax()][0])
                            leftmost_points.append(leftmost_point)
                            rightmost_points.append(rightmost_point)

                    # Iterate over each contour
                    for i, contour1 in enumerate(contours):
                        if len(contour1) >= 4:
                            leftmost_point1 = leftmost_points[i]
                            rightmost_point1 = rightmost_points[i]
                            min_distance_left = float('inf')
                            min_distance_right = float('inf')
                            closest_left_point = None
                            closest_right_point = None

                            # Iterate over each other contour
                            for j, contour2 in enumerate(contours):
                                if i != j and len(contour2) >= 4:
                                    leftmost_point2 = leftmost_points[j]
                                    rightmost_point2 = rightmost_points[j]

                                    distance_left = np.linalg.norm(
                                        np.array(leftmost_point1) - np.array(leftmost_point2))
                                    if distance_left < min_distance_left:
                                        min_distance_left = distance_left
                                        closest_left_point = leftmost_point2

                                    # Calculate distances for rightmost points
                                    distance_right = np.linalg.norm(
                                        np.array(rightmost_point1) - np.array(rightmost_point2))
                                    if distance_right < min_distance_right:
                                        min_distance_right = distance_right
                                        closest_right_point = rightmost_point2

                            # Check if the closest points are within half the frame height
                            if closest_left_point is not None and min_distance_left < frame.shape[0] / 2:
                                cv2.line(frame_mask, leftmost_point1, closest_left_point, (255, 0, 0), 5)
                                line_list.append((leftmost_point1, closest_left_point))

                            if closest_right_point is not None and min_distance_right < frame.shape[0] / 2:
                                cv2.line(frame_mask, rightmost_point1, closest_right_point, (255, 0, 0), 5)
                                line_list.append((rightmost_point1, closest_right_point))

        # cv2.imshow("frame segment mask", frame_mask)

        return frame_mask, line_list
