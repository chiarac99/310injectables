"""
undistort.py

Takes an array of points in the distorted image (with wide angle lens)
and converts each x,y pair to a real x-y coordinate in the plane of interest

Uses the calibration_data.npz file for calibration params
such as camera matrix and distortion coefficients.
"""

import numpy as np
import cv2 as cv


def load_calibration_data():
    """
    Gets all calibration data from file
    Returns: data, camera matrix, distortion coefficients, homography matrix and newcameramatx
    """

    data = np.load('calibration_data.npz')
    mtx = data['mtx']
    dist = data['dist']
    H = data['H']
    newcameramtx = data['newcameramtx']

    return data, mtx, dist, H, newcameramtx


def convert_distorted_coordinate_to_real_space(u, v):
    """
    Removes distortion for a single pair of coordinates
    :param u: x-coordinate in original image
    :param v: y-coordinate in original image
    :return: tuple (x,y) in real space
    """

    # load calibration data
    data, mtx, dist, H, newcameramtx = load_calibration_data()

    src_pt = np.array([[u, v]], dtype=np.float32)

    # tranform pixel points to real space using inverse of H matrix transform
    H_inverse = np.linalg.inv(H)
    dst_pt = cv.perspectiveTransform(src_pt[None,:,:], H_inverse)
    new_x, new_y = dst_pt[0,0]
    print(f"Point at image ({u:.1f},{v:.1f}) -> ({new_x:.2f}, {new_y:.2f}) mm")

    return (new_x, new_y)


def convert_distorted_coordinate_to_pixel_space(u, v):
    """
    Converts distorted single pair of coordinates to undistorted pixel space
    Args:
        u: x-coordinate in original image
        v: y-coordinate in original image
    Returns:
        new_coordinate: tuple
    """

    # load calibration data
    data, mtx, dist, H, newcameramtx = load_calibration_data()

    # undistort point
    undistorted_pt = cv.undistortPoints(np.array([[[u,v]]], dtype=np.float32), mtx, dist, P=newcameramtx)

    return undistorted_pt


def undistort_img(img, filename=False, display=True):
    """
    Display an undistorted image
    Args:
        img: ndarray (cv2 rendered image matrix)
        filename: str (if want to read from file instead)
        display: bool (true if display of img wanted)

    Returns:
        undistorted image matrix
    """
    if filename:
        # if reading from file, replace img with image from file
        img = cv.imread(filename)

    # load calibration data
    data, mtx, dist, H, newcameramtx = load_calibration_data()

    # undistort img
    undistorted_img = cv.undistort(img, mtx, dist, None, newcameramtx)

    if display:
        # display undistorted img
        cv.imshow("Undistorted image", undistorted_img)

    return undistorted_img
