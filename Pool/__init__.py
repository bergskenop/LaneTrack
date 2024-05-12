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
    def __init__(self, type=None):
        """
        :param type: 50m or 25m pool

        """
        self.ready = False
        self.remarkable_frames = []
        self.lane_angles = []
        self.Lanes = []
        self.type = type if type is not None else None
        self.selected_lane = None
        self.LD = Lanes.LaneDetect.LaneDetect(houghPrecisionDegree=np.pi / 720)
        self.middle_angle = None
        self.range_angle = None
        self.cf = ColorFilter.Colorfilter()
        self.swimmer = None
        self.tracker = 'botsort.yaml'
        self.model = YOLO('weights/yolov8n_swimmer_v7.pt')
        self.frame_instance = None
        # Variable lookingAt indicates whether we're looking at right side (0), middle (1) or the left side of the pool (2)
        self.lookingAt = 2

    def setup(self):
        self.ready = True

    def initialise_lanes(self, data_path, frame_indent=100, fps_div=1):
        """
        :param frame_indent: can be used to skip frames within recording, accelerating code
        :param data_path: video location
        runs through entire video analysing lane divider angles
        results will be used to estimate location within swimming pool
        """
        print("\nLooping video to analyse reference position")
        for idx, frame in enumerate(read_video(data_path, frame_skip=frame_indent)):
            self.LD.append(frame, pos=int(idx * frame_indent / fps_div))

        data_array = np.nan_to_num(self.LD.get_median_rotation())
        plt.title("Not interpolated perceived lane angles")
        plt.xlabel("Frame index")
        plt.ylabel("Rotation (degrees)")
        plt.plot(data_array)
        plt.show()

        self.LD.interpolate_nan()
        plt.title("Interpolated perceived lane angles")
        plt.xlabel("Frame index")
        plt.ylabel("Rotation (degrees)")
        plt.plot(self.LD.get_median_rotation())
        plt.show()
        self.LD.get_filtered_list()
        max_angle = max(self.LD.get_median_rotation())
        min_angle = min(self.LD.get_median_rotation())
        self.range_angle = max_angle + abs(min_angle)
        self.middle_angle = max_angle - self.range_angle / 2

        plt.title("Interpolated and filtered perceived lane angles")
        plt.xlabel("Frame index")
        plt.ylabel("Rotation (degrees)")
        plt.plot(self.LD.get_filtered_list())
        plt.show()

    def process(self, data_path, fps_div=1):
        """
        :param data_path:
        creates object Fram, Swimmer, Lane
        Yields annotated image for visualisation
        """
        data_type = check_path_type(data_path)
        if data_type == "video":
            for idx, frame in enumerate(read_video(data_path, frame_skip=fps_div)):
                # region PoolRegionDecision
                # Describes where the camera is pointed at 0 being right side, 1 middle and 2 left side.
                if len(self.LD.get_median_rotation()) >= idx:
                    if self.middle_angle - self.range_angle / 4 < self.LD.rotation_list_median[idx-1] < self.middle_angle \
                            + self.range_angle / 4:
                        self.lookingAt = 1
                    elif self.LD.rotation_list_median[idx-1] > self.middle_angle + self.range_angle / 4:
                        self.lookingAt = 2
                    elif self.LD.rotation_list_median[idx-1] < self.middle_angle - self.range_angle / 4:
                        self.lookingAt = 0
                # endregion
                # region SelectLaneAndSwimmer
                # Allows user to select lane and swimmer
                if self.selected_lane is None:
                    self.select_lane(frame)
                    self.swimmer = Swimmer.Swimmer()
                elif self.swimmer.id is None:
                    self.select_swimmer(frame)
                    self.frame_instance = Frame.Frame(frame, self.model, self.cf, self.LD, self.selected_lane,
                                                      self.lookingAt, self.swimmer.id)
                    print(f'Tracking swimmer with id: {self.swimmer.id}')
                # endregion
                # region TrackAndDetectSegmentCrossing
                # Adds new frame to frame object, passes the detection model alongside the lookingAT variable
                # If a frame is remarkable i.e. crossing a segment it gets added to the frame list for further use.
                else:
                    ret = self.frame_instance.add_frame(frame, self.model, self.lookingAt)
                    if ret:
                        if self.frame_instance.remarkable:
                            self.remarkable_frames.append((idx, self.frame_instance))
                        annotated_image = put_text_on_image(self.frame_instance.filled_image,
                                                            self.frame_instance.state_verbose, self.frame_instance.state)
                        display(annotated_image)
                        cv2.waitKey(5)
                        yield annotated_image
                # endregion
        # region ImageAndDirectoryHandling
        elif data_type == 'image':
            img = cv2.imread(data_path)
        elif data_type == 'directory':
            for filename in os.listdir(data_path):
                image_path = os.path.join(data_path, filename)
                img = cv2.imread(image_path)
        # endregion

    # region SelectLaneAndSwimmer
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
            cv2.putText(filled_image, "Select swimming lane to track", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 0), 2)
            cv2.imshow('Contour Selection', filled_image)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cv2.destroyWindow('Contour Selection')
        if valid_contours is not None:
            self.Lanes = valid_contours
            self.selected_lane = Lanes.Lane(frame, valid_contours[selected_contour_index])

        print(f"Swimming pool has been initiated with {len(self.Lanes)} visible lanes.")

    def select_swimmer(self, frame):
        def draw_boxes(image, xyxys, class_ids, hovered_id=None, prev_hovered_id=None):
            for xyxy, class_id, object_id in zip(xyxys, class_ids, object_ids):
                if class_id == 0.0:
                    continue
                color = (0, 255, 0)
                thickness = 2
                if hovered_id is not None and object_id == hovered_id:
                    color = (0, 0, 255)
                    thickness = 3
                elif prev_hovered_id is not None and object_id == prev_hovered_id:
                    color = (0, 255, 0)
                    thickness = 2
                cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, thickness)
                cv2.putText(image, f'ID: {object_id}', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            color, 2)

        # Mouse callback function
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                hovered_id = None
                for xyxy, class_id, object_id in zip(param['xyxys'], param['class_ids'], param['object_ids']):
                    if class_id == 0.0:
                        continue
                    if xyxy[0] < x < xyxy[2] and xyxy[1] < y < xyxy[3]:
                        hovered_id = object_id
                        break
                if hovered_id != param['hovered_id']:
                    draw_boxes(param['image'], param['xyxys'], param['class_ids'], hovered_id, param['hovered_id'])
                    param['hovered_id'] = hovered_id
                    cv2.imshow("Image", param['image'])
            elif event == cv2.EVENT_LBUTTONDOWN:
                for xyxy, object_id in zip(param['xyxys'], param['object_ids']):
                    if xyxy[0] < x < xyxy[2] and xyxy[1] < y < xyxy[3]:
                        print(f"Clicked on box with ID: {object_id}")
                        self.swimmer.id = object_id
                        cv2.destroyAllWindows()
                        return

        initial_results = self.model.track(frame, tracker=self.tracker, verbose=True,
                                           persist=True, conf=0.05, iou=0.2, augment=True,
                                           agnostic_nms=True)
        for result in initial_results:
            frame_cpy = frame.copy()
            boxes = result.boxes.cpu().numpy()
            xyxys = boxes.xyxy
            class_ids = boxes.cls
            object_ids = boxes.id

            draw_boxes(frame_cpy, xyxys, class_ids)
            cv2.namedWindow("Image", cv2.WINDOW_KEEPRATIO)
            cv2.setMouseCallback("Image", mouse_callback,
                                 {'image': frame_cpy, 'xyxys': xyxys, 'class_ids': class_ids, 'object_ids': object_ids,
                                  'hovered_id': None})
            cv2.imshow("Image", frame_cpy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # endregion
