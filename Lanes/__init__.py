from helper import *
import ColorFilter

class Lane:
    def __init__(self, frame, contour, cf=ColorFilter.Colorfilter()):
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        self.cf = cf
        self.lane_n = None
        self.frame = frame
        self.lane_mask = np.zeros_like(frame)
        self.contour_area = cv2.contourArea(contour)
        cv2.drawContours(self.lane_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        self.red_lines = []
        self.visual_lane = None

    def track_lane(self, frame):
        frame_resize = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frame_cpy = frame.copy()

        _, lane_mask = self.cf.get_lanes(frame_resize, redMode=False)
        border_mask = self.cf.swimming_pool_box(frame_resize)[0]

        if border_mask is not None:
            mask = cv2.bitwise_or(lane_mask, border_mask)
            mask = cv2.bitwise_not(mask)
        else:
            mask = cv2.bitwise_not(lane_mask)

        mask = cv2.resize(mask, None, fx=2, fy=2)

        contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = validate_contours(frame, contours, self.contour_area)
        cv2.drawContours(frame_cpy, valid_contours, -1, (255, 0, 255), thickness=cv2.FILLED)

        if valid_contours is not None:
            for contour in valid_contours:
                frame_mask_cpy = frame.copy()
                new_mask = np.zeros_like(frame_mask_cpy)
                cv2.drawContours(new_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                alpha = 0.25
                overlap = cv2.bitwise_and(new_mask, self.lane_mask)

                overlap_percentage = cv2.countNonZero(overlap[:, :, 0])/cv2.countNonZero(self.lane_mask[:, :, 0])
                if overlap_percentage > 0.4:
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
