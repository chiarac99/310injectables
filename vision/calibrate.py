"""
Calibration.py

Calibrate once (including homography matrix determination), use 10-20 images for robust calibration

Notes:
-   Calibration should be done with a chessboard (see pdf in the repo).
    This chessboard must be the same dimensions as the pattern_size param defined at the start of the doc with squares accurately measured.
    Pattern size is defined as the number of columns vs. rows of chessboard "corners" (where four black and white checkers meet).
    Squares should be measured as an average over the entire board.
    Mount printed chessboard to foamcore to ensure that all dimensions are good

-   Save all calibration images to a folder named: checkboard_images/

-   Mount camera such that lens is parallel to the plane where syringe will be located. The camera should stay stationary during calibration.
    The distance between lens and plane is not embedded in the code. It can change.
    But if the distance between the camera and moving step is changed in the design, be sure to calibrate the camera for this new distance.

-   Take photos of the chessboard flat on the surface and different orientations in x/y e.g. centred, to the left/right, at the top/bottom, at slight angles.
    The chessboard must always be at the orientation of pattern_size (i.e. do not turn it horizontal if the pattern_size is portrait).
    The entire chessboard must be within the frame.

Saves to a file 'calibration_data.npz':
    - camera matrix
    - distortion parameters (removes distortion in pixel space)
    - homography matrix (converts distorted pixel space to real space)

"""

import numpy as np
import cv2 as cv
import glob

import undistort

# define chessboard pattern aspect ratio and square size
pattern_size = (17, 6)  # (columns, rows) = (x, y) in OpenCV
square_size = 14.395  # mm


def calibrate():
    """

    Returns:

    """

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp[:, :2] *= square_size  # scale to real-world units (mm)

    # arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    # loop through all images in dir to calibrate
    images = glob.glob("checkboard_images/*.jpg")
    images_processed = (
        []
    )  # for recording the filenames of the images that are successfully processed

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # grayscale the image
        cv.imshow("img", gray)
        # find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

        # if found, add object points, image points (after refining them)
        if (
            ret == True
            and corners is not None
            and len(corners) == pattern_size[0] * pattern_size[1]
        ):
            images_processed.append(
                fname
            )  # saves names of files that were actually able to be processed
            objpoints.append(objp)

            # refines corners to subpixel accuracy
            # 11 x 11 pixels window for refinement strikes balance btw accuracy and efficiency
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # draw and display the corners
            cv.drawChessboardCorners(img, pattern_size, corners2, ret)

            # display (if necessary)
            cv.imshow("img", img)
            cv.waitKey(500)  # only display for 500ms

    cv.destroyAllWindows()  # close all show windows

    # get calibration info
    # returns camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # get H homography matrix
    # choose the first image's points to compute homography
    # use only x and y (ignore z since it's 0)
    objp_2d = objpoints[0][:, :2]  # shape: (63, 2), real-world 2D points
    imgp_2d = imgpoints[0].reshape(-1, 2)  # shape: (63, 2), pixel locations

    # compute homography from object (real) points to image points
    H, _ = cv.findHomography(objp_2d, imgp_2d)
    # flatten the list of arrays into a single array
    all_objpoints = np.concatenate(objpoints, axis=0)  # shape (N, 2)
    all_imgpoints = np.concatenate(imgpoints, axis=0)  # shape (N, 2)

    # # compute the best-fit homography using all the correspondences
    # H, mask = cv.findHomography(all_objpoints, all_imgpoints, method=cv.RANSAC)

    # load an image from the calibration set for computing newcameramtx
    # TODO: this newcameramtx just comes from a single calibration image??
    img_no = 0
    fname = images_processed[img_no]
    img = cv.imread(fname)
    h, w = img.shape[:2]

    # Compute new camera matrix to minimize black borders
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Save initial bounding box - this will need to be updated with every calibration
    bounding_box = (454, 458, 454 + 1234, 458 + 133)

    # save initial guess at pixel to step ratio
    pixel_to_step_ratio = 112.4

    # initialise step_gap_h
    step_gap_h = 0

    # save calibration data to file
    np.savez(
        "calibration_data.npz",
        mtx=mtx,
        dist=dist,
        H=H,
        newcameramtx=newcameramtx,
        bounding_box=bounding_box,
        pixel_to_step_ratio=pixel_to_step_ratio,
        step_gap_h=step_gap_h,
    )

    # -------------------
    # FOR VISUALISATION
    # see the distorted image next to the undistorted image
    # with two corners compared
    # change which of the calibration images you're looking at by changing the param im_no
    # -------------------
    # undistort the image
    undistorted_img = cv.undistort(img, mtx, dist, None, newcameramtx)

    # resize images to be the same height for stacking, if necessary
    if img.shape != undistorted_img.shape:
        undistorted_img = cv.resize(undistorted_img, (img.shape[1], img.shape[0]))

    # grab 2 corners
    corner1 = imgpoints[img_no][0][0]
    corner2 = imgpoints[img_no][1][0]
    # undistort two corners
    undist1 = cv.undistortPoints(
        np.array([[corner1]], dtype=np.float32), mtx, dist, P=newcameramtx
    )
    undist2 = cv.undistortPoints(
        np.array([[corner2]], dtype=np.float32), mtx, dist, P=newcameramtx
    )
    x_1, y_1 = undist1[0, 0]
    x_2, y_2 = undist2[0, 0]

    # display distorted dots on original image
    cv.circle(
        img,
        (int(corner1[0]), int(corner1[1])),
        radius=5,
        color=(0, 0, 255),
        thickness=-1,
    )  # red dot
    cv.circle(
        img,
        (int(corner2[0]), int(corner2[1])),
        radius=5,
        color=(255, 0, 0),
        thickness=-1,
    )  # blue dot

    # display undistorted dots on undistorted image
    cv.circle(
        undistorted_img, (int(x_1), int(y_1)), radius=5, color=(0, 0, 255), thickness=-1
    )  # red dot
    cv.circle(
        undistorted_img, (int(x_2), int(y_2)), radius=5, color=(255, 0, 0), thickness=-1
    )  # blue dot

    # tranform distorted pixel points to real space using H matrix transform
    src_pt_1 = np.array([[[corner1[0], corner1[1]]]], dtype=np.float32)
    src_pt_2 = np.array([[[corner2[0], corner2[1]]]], dtype=np.float32)
    H_inverse = np.linalg.inv(H)
    dst_pt_1 = cv.perspectiveTransform(src_pt_1, H_inverse)
    dst_pt_2 = cv.perspectiveTransform(src_pt_2, H_inverse)
    pt1 = dst_pt_1[0, 0]
    pt2 = dst_pt_2[0, 0]
    print(f"distance btw points = {distance_between_points(pt1, pt2)}")

    # stack side-by-side and display
    combined = np.hstack((img, undistorted_img))
    cv.imshow("Original (Left) vs Undistorted (Right)", combined)
    cv.waitKey(0)  # press 0 to close
    cv.destroyAllWindows()


def distance_between_points(p1, p2):
    """
    Computes the Euclidean distance between two points in 2D.

    Args:
        p1: tuple or array-like of (x1, y1)
        p2: tuple or array-like of (x2, y2)

    Returns:
        float: distance between the points
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)


def load_calibration_data():
    """
    Load existing calibration data
    Returns: mtx, dist, H, newcameramtx
    """
    data = np.load("calibration_data.npz")

    # Step 2: Extract existing data
    mtx = data["mtx"]
    dist = data["dist"]
    H = data["H"]
    newcameramtx = data["newcameramtx"]
    bounding_box = data["bounding_box"]
    pixel_to_step_ratio = data["pixel_to_step_ratio"]
    step_gap_h = data["step_gap_h"]

    return mtx, dist, H, newcameramtx, bounding_box, pixel_to_step_ratio, step_gap_h


def redo_bounding_box():
    """
    Redefine bounding box
    1. Take image using connected camera
    2. Allow user to select bounding box
    3. Update bounding box in calibration file
    """

    # open camera
    camera = cv.VideoCapture(0)
    # warm up camera
    for _ in range(5):
        # take image
        ret, latest_frame = camera.read()

    # undistort frame
    undistorted_frame = (
        latest_frame  # undistort.undistort_img(latest_frame, display=False)
    )

    # select bounding box
    # First selection
    r = cv.selectROI(
        "Select Bounding Box for entire step",
        undistorted_frame,
        fromCenter=False,
        showCrosshair=True,
    )
    print("\nSelected ROI (x, y, width, height):", r)
    # Do NOT destroy windows here

    # Convert bounding box coordinates to integers and ensure they're within image bounds
    # find bounding box (x1, y1, x2, y2) for segment.py
    new_bounding_box = (r[0], r[1], r[0] + r[2], r[1] + r[3])
    print("New bounding box:", new_bounding_box)
    h, w = undistorted_frame.shape[:2]
    x1, y1, x2, y2 = [int(coord) for coord in new_bounding_box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Crop image using validated coordinates
    new_img = undistorted_frame[y1:y2, x1:x2]

    # Second selection for step gap
    # TODO: check this works!
    r2 = cv.selectROI(
        "Select Bounding Box for step gap",
        new_img,
        fromCenter=False,
        showCrosshair=True,
    )

    # third selection for blade positions
    # List to store the clicked points
    points = []

    # Mouse callback function to record the clicks
    def click_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            # Draw a small circle where the user clicked
            cv.circle(undistorted_frame, (x, y), 5, (0, 0, 255), -1)
            cv.imshow("Click in two points where the blades are located", undistorted_frame)

    cv.imshow("Click in two points where the blades are located", undistorted_frame)
    cv.setMouseCallback("Click in two points where the blades are located", click_event)

    print("Click two points on the image where the two blades are located...")
    print("Press 'c' to exit")

    # Wait until two points are clicked
    while True:
        key = cv.waitKey(1) & 0xFF
        if len(points) == 2:
            print("Two points selected.")
            break
        if key == 'c':
            break

    cv.destroyAllWindows()

    # Access the coordinates
    if len(points) == 2:
        blade_xs = [points[0][0], points[1][0]]
        bladeL_pixels_x = min(blade_xs)
        bladeR_pixels_x = max(blade_xs)

        print("BladeL x:", bladeL_pixels_x)
        print("BladeR x:", bladeR_pixels_x)

    cv.destroyAllWindows()  # Destroy windows only once after both selections

    # update bounding box
    mtx, dist, H, newcameramtx, old_bounding_box, pixel_to_step_ratio, step_gap_h = (
        load_calibration_data()
    )
    print("Old step gap:", old_bounding_box)

    # pixel to length ratio from bounding box
    step_real_width = 11  # inches
    new_ratio = r[2] / step_real_width

    # now edit step gap height
    # select bounding box
    new_step_gap_h = r2[3]
    print("\nSelected step gap :", r2[3])

    # save updated data back
    np.savez(
        "calibration_data.npz",
        mtx=mtx,
        dist=dist,
        H=H,
        newcameramtx=newcameramtx,
        bounding_box=new_bounding_box,
        pixel_to_step_ratio=new_ratio,
        step_gap_h=new_step_gap_h,
    )

    # Verify the save
    data = np.load("calibration_data.npz")
    print("Verified saved bounding box:", data["bounding_box"])
    print("Verified saved step gap:", data["step_gap_h"])
    print("\nBounding box updated successfully!")


def generate_new_calibration_file():
    """
    Generates empty calibration file
    """
    np.savez(
        "calibration_data.npz",
        mtx=0,
        dist=0,
        H=0,
        newcameramtx=0,
        bounding_box=0,
        pixel_to_step_ratio=0,
        step_gap_h=0,
    )


# calibrate()
generate_new_calibration_file()
redo_bounding_box()
