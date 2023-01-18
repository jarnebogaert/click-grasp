import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.reprojection import reproject_to_frame_z_plane
from airo_camera_toolkit.utils import ImageConverter
from airo_robots.grippers.hardware.robotiq_2f85_tcp import Robotiq2F85
from airo_robots.manipulators.hardware.ur_rtde import UR3E_CONFIG, UR_RTDE

# from rtde_control import RTDEControlInterface


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


def draw_pose(image, pose, world_to_camera, camera_matrix):
    pose_camera = world_to_camera @ pose
    rvec = pose_camera[:3, :3]
    tvec = pose_camera[:3, -1]
    image = cv2.drawFrameAxes(image, camera_matrix, np.zeros(4), rvec, tvec, 0.05)
    return image


def make_grasp_pose(clicked_points):
    grasp_location = (clicked_points[1] + clicked_points[0]) / 2

    # Build the orientation matrix so that the gripper opens along the line between the clicked points.
    gripper_open_direction = clicked_points[1] - clicked_points[0]
    X = gripper_open_direction / np.linalg.norm(gripper_open_direction)
    Z = np.array([0, 0, -1])  # topdown
    Y = np.cross(Z, X)
    grasp_orientation = np.column_stack([X, Y, Z])

    # Assemble the 4x4 pose matrix
    grasp_pose = np.identity(4)
    grasp_pose[:3, -1] = grasp_location
    grasp_pose[:3, :3] = grasp_orientation
    return grasp_pose


if __name__ == "__main__":  # noqa: C901
    if not os.path.exists(Path(__file__).parent / "marker.pickle"):
        print("Please run camera_calibration.py first.")
        sys.exit(0)
    world_to_camera = load_saved_calibration()

    ip_louise = "10.42.0.163"
    ur3e = UR_RTDE(ip_louise, UR3E_CONFIG)

    gripper = Robotiq2F85(ip_louise)

    ur3e_in_world = np.identity(4)
    ur3e_in_world[:3, -1] += [0.3, 0, 0]  # 30 cm positive along x from where the marker should be placed
    world_to_ur3e = np.linalg.inv(ur3e_in_world)
    # move_speed = 0.1  # m /s

    home_ur3e = ur3e_in_world.copy()
    home_ur3e[:3, -1] += [-0.15, -0.20, 0.2]
    X = np.array([1, 0, 0])
    Z = np.array([0, 0, -1])  # topdown
    Y = np.cross(Z, X)
    top_down = np.column_stack([X, Y, Z])
    home_ur3e[:3, :3] = top_down
    home_in_ur3e = world_to_ur3e @ home_ur3e

    ur3e.move_linear_to_tcp_pose(home_in_ur3e)

    gripper.open()

    zed = Zed2i(resolution=Zed2i.RESOLUTION_720, fps=30)
    intrinsics_matrix = zed.intrinsics_matrix

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

    grasp_executed = False

    while True:
        _, h, w = zed.get_rgb_image().shape
        image = zed.get_rgb_image()
        image = ImageConverter(image).image_in_opencv_format
        image = draw_clicked_grasp(image, clicked_image_points, current_mouse_point)

        draw_pose(image, ur3e_in_world, world_to_camera, intrinsics_matrix)

        if len(clicked_image_points) == 2:
            points_in_image = np.array(clicked_image_points)
            # points_in_world = reproject_to_frame_z_plane(points_in_image, intrinsics_matrix, world_to_camera)
            points_in_world = reproject_to_frame_z_plane(
                points_in_image, intrinsics_matrix, np.linalg.inv(world_to_camera)
            )

            grasp_pose = make_grasp_pose(points_in_world)
            grasp_pose[2, -1] += 0.005
            grasp_pose_in_ur3e = world_to_ur3e @ grasp_pose

            pregrasp_pose = np.copy(grasp_pose)
            pregrasp_pose[2, -1] += 0.15  # raise grasp height by 15 cm for safety
            pregrasp_pose_in_ur3e = world_to_ur3e @ pregrasp_pose

            image = draw_pose(image, grasp_pose, world_to_camera, intrinsics_matrix)
            image = draw_pose(image, pregrasp_pose, world_to_camera, intrinsics_matrix)

            if not grasp_executed:
                cv2.imshow(window_name, image)  # refresh image

                ur3e.move_linear_to_tcp_pose(pregrasp_pose_in_ur3e)
                ur3e.move_linear_to_tcp_pose(grasp_pose_in_ur3e)
                gripper.close()
                ur3e.move_linear_to_tcp_pose(pregrasp_pose_in_ur3e)
                grasp_executed = True

        cv2.imshow(window_name, image)
        key = cv2.waitKey(10)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break
