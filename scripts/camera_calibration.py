import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
from camera_toolkit.aruco import get_aruco_marker_poses
from camera_toolkit.reproject import project_world_to_image_plane
from camera_toolkit.zed2i import Zed2i


class CameraMapping:
    serial_tripod = 38633712
    serial_side = 35357320
    serial_top = 31733653


def draw_center_circle(image) -> np.ndarray:
    h, w, _ = image.shape
    center_u = w // 2
    center_v = h // 2
    center = (center_u, center_v)
    image = cv2.circle(image, center, 1, (255, 0, 255), thickness=2)
    return image


def draw_world_axes(image, world_to_camera, camera_matrix):
    origin = project_world_to_image_plane(np.zeros(3), world_to_camera, camera_matrix).astype(int)
    image = cv2.circle(image, origin.T, 10, (0, 255, 255), thickness=2)

    x_pos = project_world_to_image_plane([1.0, 0.0, 0.0], world_to_camera, camera_matrix).astype(int)
    x_neg = project_world_to_image_plane([-1.0, 0.0, 0.0], world_to_camera, camera_matrix).astype(int)
    y_pos = project_world_to_image_plane([0.0, 1.0, 0.0], world_to_camera, camera_matrix).astype(int)
    y_neg = project_world_to_image_plane([0.0, -1.0, 0.0], world_to_camera, camera_matrix).astype(int)
    image = cv2.line(image, x_pos.T, origin.T, color=(0, 0, 255), thickness=2)
    image = cv2.line(image, x_neg.T, origin.T, color=(100, 100, 255), thickness=2)
    image = cv2.line(image, y_pos.T, origin.T, color=(0, 255, 0), thickness=2)
    image = cv2.line(image, y_neg.T, origin.T, color=(150, 255, 150), thickness=2)

    z_pos = project_world_to_image_plane([0.0, 0.0, 1.0], world_to_camera, camera_matrix).astype(int)
    image = cv2.line(image, z_pos.T, origin.T, color=(255, 0, 0), thickness=2)
    return image


def save_calibration(rotation_matrix, translation):
    with open(Path(__file__).parent / "marker.pickle", "wb") as f:
        pickle.dump([translation, rotation_matrix], f)


if __name__ == "__main__":
    resolution = sl.RESOLUTION.HD720
    zed = Zed2i(resolution=resolution, serial_number=CameraMapping.serial_top, fps=30)

    # Configure custom project-wide InputTransform based on camera, resolution, etc.
    _, h, w = zed.get_rgb_image().shape

    print("Press s to save Marker pose, q to quit.")
    while True:
        start_time = time.time()
        image = zed.get_rgb_image()
        image = zed.image_shape_torch_to_opencv(image)
        image = image.copy()
        cam_matrix = zed.get_camera_matrix()
        image, translations, rotations, _ = get_aruco_marker_poses(
            image, cam_matrix, 0.106, cv2.aruco.DICT_6X6_250, True
        )
        image = draw_center_circle(image)

        if rotations is not None:
            aruco_in_camera_transform = np.eye(4)
            aruco_in_camera_transform[:3, :3] = rotations[0]
            aruco_in_camera_transform[:3, 3] = translations[0]
            world_to_camera = aruco_in_camera_transform
            camera_matrix = zed.get_camera_matrix()
            image = draw_world_axes(image, world_to_camera, camera_matrix)

        cv2.imshow("Camera feed", image)

        key = cv2.waitKey(10)
        if key == ord("s") and rotations is not None:
            print("Saving current marker pose to pickle.")
            save_calibration(rotations[0], translations[0])
            time.sleep(5)
        if key == ord("q"):
            cv2.destroyAllWindows()
            zed.close()
            break
