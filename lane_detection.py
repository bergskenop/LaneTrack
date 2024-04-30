import cv2
import numpy as np
from helper import *

import ColorFilter

cf = ColorFilter.Colorfilter()

video_path='Fanny Lecluyse 50m SS Reeksen EK25m 2015_1 - Swimming.avi'

for idx, frame in enumerate(read_video(video_path)):
    end_start_mask, val = cf.swimming_pool_box(frame)
    lanes_list, image_with_colored_space, frame_lanes_mask = get_lanes(frame)
    display(cv2.bitwise_or(frame_lanes_mask, end_start_mask))
    display(image_with_colored_space, 'result')
    cv2.waitKey(5)


