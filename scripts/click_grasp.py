import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
from camera_toolkit.zed2i import Zed2i


class CameraMapping:
    serial_tripod = 38633712
    serial_side = 35357320
    serial_top = 31733653


def load_saved_calibration():
    with open(Path(__file__).parent / "marker.pickle", "rb") as f:
        aruco_in_camera_position, aruco_in_camera_orientation = pickle.load(f)
    # get camera extrinsics transform
    aruco_in_camera_transform = np.eye(4)
    aruco_in_camera_transform[:3, :3] = aruco_in_camera_orientation
    aruco_in_camera_transform[:3, 3] = aruco_in_camera_position
    return aruco_in_camera_transform


def draw_clicked_grasp(image, clicked_image_points, current_mouse_point):
    """If we don't have tow clicks yet, draw a line between the first point and the current cursor position."""
    for point in clicked_image_points:
        image = cv2.circle(image, point, 5, (0, 255, 0), thickness=2)

    if len(clicked_image_points) >= 1:
        first_point = clicked_image_points[0]
        second_point = current_mouse_point[0]

        if len(clicked_image_points) >= 2:
            second_point = clicked_image_points[1]

        image = cv2.line(image, first_point, second_point, color=(0, 255, 0), thickness=1)
        middle = (np.array(first_point) + np.array(second_point)) // 2
        image = cv2.circle(image, middle.T, 2, (0, 255, 0), thickness=2)

    return image


if __name__ == "__main__":
    if not os.path.exists(Path(__file__).parent / "marker.pickle"):
        print("Please run camera_calibration.py first.")
        sys.exit(0)
    world_to_camera = load_saved_calibration()

    resolution = sl.RESOLUTION.HD720
    zed = Zed2i(resolution=resolution, serial_number=CameraMapping.serial_top, fps=30)

    current_mouse_point = [(0, 0)]  # has to be a list so that the callback can edit it
    clicked_image_points = []

    def mouse_callback(event, x, y, flags, parm):
        if len(clicked_image_points) >= 2:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_image_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            current_mouse_point[0] = x, y

    window_name = "Camera feed"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        _, h, w = zed.get_rgb_image().shape
        image = zed.get_rgb_image()
        image = zed.image_shape_torch_to_opencv(image)
        image = image.copy()
        # cam_matrix = zed.get_camera_matrix()

        image = draw_clicked_grasp(image, clicked_image_points, current_mouse_point)

        cv2.imshow(window_name, image)
        key = cv2.waitKey(10)

        if key == ord("q"):
            cv2.destroyAllWindows()
            zed.close()
            break
