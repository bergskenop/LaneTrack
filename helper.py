import os
import random

import cv2
import numpy as np
import torch

import ColorFilter

def read_video(video_path: str, frame_skip: int = 1):
    """
    Read video and yield frames at specified intervals.

    Args:
        video_path (str): Path to the video file.
        frame_skip (int, optional): Number of frames to skip between reads. Default is 1.

    Yields:
        frame: A frame from the video.
    """
    print(f'\nProcessing video: {video_path.partition("/")[-1]}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file. Please check the file path or format.")

    start_frame_n = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_n)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame_n
    print(f"Total frames: {total_frames}")

    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_n % frame_skip == 0:
            progress = int(50 * frame_n / total_frames)
            percentage = round(100 * frame_n / total_frames)
            bar = f"[{'█' * progress}{' ' * (50 - progress)}] {percentage}%"
            print(f'\rProcessing footage {bar}', end='', flush=True)
            yield frame
        frame_n += 1

    cap.release()


def display(img, title='image', max_size=500000):
    """
    resizes the image before it displays it,
    this stops large stitches from going over the screen!
    """
    scale = np.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)


def validate_contours(frame, contours, prev_contour_area=200):
    valid_contours = []
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 0, 0])
    upper_blue = np.array([110, 255, 255])

    contours = [contour for contour in contours if cv2.contourArea(contour) > prev_contour_area * 0.5]

    for contour in contours:
        mask = cv2.drawContours(np.zeros_like(frame), [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(mask, hsv_frame)

        contour_pixels = cv2.countNonZero(mask[:, :, 0])
        hsv_result_mask = cv2.inRange(result, lower_blue, upper_blue)
        blue_pixels = cv2.countNonZero(hsv_result_mask)

        if blue_pixels / contour_pixels >= 0.5:
            valid_contours.append(contour)

    if valid_contours:
        area_thresh = max(cv2.contourArea(contour) for contour in valid_contours) * 0.15
        valid_contours = [contour for contour in valid_contours if cv2.contourArea(contour) > area_thresh]
        return valid_contours

    else:
        # print('No valid contours found')
        return None


def check_path_type(path):
    if os.path.isfile(path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.MTS']
        if any(path.endswith(ext) for ext in video_extensions):
            return "video"

        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        if any(path.endswith(ext) for ext in image_extensions):
            return "image"

    elif os.path.isdir(path):
        return "directory"

    return "unknown"

def part_of_swimmer(cx, cy, swimmer_box):
    xmin, ymin, xmax, ymax = swimmer_box
    return xmin <= cx <= xmax and ymin <= cy <= ymax


def box_intersects_line(box, line):
    box_left, box_top, box_right, box_bottom = box[0, 0, :]
    (line_x1, line_y1), (line_x2, line_y2) = line

    if (min(line_x1, line_x2) <= box_right and max(line_x1, line_x2) >= box_left and
            min(line_y1, line_y2) <= box_bottom and max(line_y1, line_y2) >= box_top):
        return True

    return False


def box_intersects_any_line(swimmer_box, lines, cap_box=None):
    box = swimmer_box if cap_box is None else cap_box
    for line in lines:
        if box_intersects_line(box, line):
            return True
    return False


def draw_box_on_image(frame, box, color=(255, 0, 0), thickness=4):
    left, top, right, bottom = box[0, 0, :]
    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, thickness)
    return frame

def has_overlap_with_other_boxes(predefined_box, all_boxes):
    for box in all_boxes:
        # Check if it's the same box
        if torch.equal(box, predefined_box):
            continue

        # Extract coordinates of the box
        box_left, box_top, box_right, box_bottom = box
        predefined_left, predefined_top, predefined_right, predefined_bottom = predefined_box[0, 0]

        # Check for overlap
        if (box_left < predefined_right and box_right > predefined_left and
                box_top < predefined_bottom and box_bottom > predefined_top):
            return box  # Overlapping box found

    return None  # No overlapping box found


def put_text_on_image(frame, text, state):
    # Define text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1

    if state == 0:  # Swimmer detected and tracking
        font_color = (0, 255, 0)  # Green
    elif state == 1:  # Swimmer crossing segment
        font_color = (255, 0, 255)  # Purple
    else:  # Swimmer lost
        font_color = (255, 0, 0)  # Red

    thickness = 2
    text_org = (10, 30)  # Position of the first line of text (top left corner)

    # Put text on the image
    for i, line in enumerate(text):
        if i > 0:
            text_org = (text_org[0], text_org[1] + 30)  # Increment y-coordinate for subsequent lines
        cv2.putText(frame, line, text_org, font, font_scale, font_color, thickness)

    return frame

