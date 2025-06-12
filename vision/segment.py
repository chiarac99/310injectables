"""
Segment.py: Image Segmentation and Plunger Detection Module

This module provides functionality for:
1. Segmenting syringe images using SAM (Segment Anything Model)
2. Detecting the position of the black plunger
3. Calculating cutting positions based on syringe orientation

Author: The Injectables | ME310
"""

import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import undistort
import matplotlib.pyplot as plt

# from vision.undistort import undistort_img
# Constants
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
# SAM_CHECKPOINT = "/Users/venkatasaisarangrandhe/sam_weights/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

# Hardware Constants (in inches)
PADDLE_THICKNESS = 0.1
STEPS_TO_LENGTH_RATIO = 1458 / 11.48  # 135.5  # 200 steps/rev, 0.47" diameter pulley
MAX_PADDLE_POS = 11.48  # inches / TODO: update this to actual length when paddle extrusion is replaced
MAX_NEEDLE_LEN = 3.6  # inches / maximum length of needle part we can process, this length is equal to the distance between a blade and the side sheet (i.e. side channel width)
# inches


def segment(img):
    """
    Segment the syringe in the input image using SAM model.

    Args:
        img:

    Returns:
        tuple: (
            image_rgb (np.ndarray): RGB version of input image,
            orientation (str): Syringe orientation ('L' or 'R'),
            mask (np.ndarray): Binary segmentation mask,
            syringe_start_col (int): Starting column of syringe
        )
    """
    # Initialize SAM model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Apply bounding box from calibration
    _, _, _, _, _, bounding_box, _, step_gap_h, _, _ = undistort.load_calibration_data()

    # Convert bounding box coordinates to integers and ensure they're within image bounds
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(coord) for coord in bounding_box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Crop image using validated coordinates
    img = img[y1:y2, x1:x2]

    # Convert to RGB for processing
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_rgb, cmap='gray')
    # plt.title("Cropped image")
    # plt.axis('off')
    # plt.show()
    # Generate prediction mask
    predictor.set_image(image_rgb)
    x, y, z = np.shape(image_rgb)
    input_box = np.array([0, 0, y, x])
    masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)

    # Clean up the mask
    mask = ~masks[0]  # Invert mask as sometimes background is detected as foreground

    # Remove step gap rows from mask
    h, w = mask.shape[:2]
    y1 = int(step_gap_h)
    y2 = h
    cropped_mask = mask[y1:y2, :]
    restored_mask = np.zeros_like(mask)
    restored_mask[y1:y2, :] = cropped_mask
    mask = restored_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # plt.imshow(mask, cmap='gray')
    # plt.title("Cropped image")
    # plt.axis('off')
    # plt.show()

    # Extract largest component to remove noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask)

    # Check if any components were found
    if num_labels <= 1:  # Only background
        print("No syringe detected - no components found")
        return image_rgb, None, np.zeros_like(mask), None, None

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_label).astype(np.uint8)

    # Check if object is present (based on area)
    syringe_area = stats[largest_label, cv2.CC_STAT_AREA]
    if syringe_area < 5000:  # Threshold may need tuning
        print("No syringe detected - area too small")
        return image_rgb, None, np.zeros_like(mask), None, None

    # Analyze mask properties
    column_sums = np.sum(mask, axis=0)
    max_width_col = np.argmax(column_sums)
    max_width_value = column_sums[max_width_col]

    print(f"max width value = {max_width_value}")

    # Check if max width is too small
    if max_width_value < 20:  # Adjust threshold as needed
        print("No syringe detected - width too small")
        return image_rgb, None, np.zeros_like(mask), None, None

    # Determine syringe orientation
    x_foreground = np.any(mask, axis=0)
    x_coords = np.where(x_foreground)[0]

    if len(x_coords) == 0:
        print("No syringe detected - no foreground pixels")
        return image_rgb, None, np.zeros_like(mask), None, None

    x_center = int(np.mean(x_coords))

    if max_width_col < x_center:
        orientation = "R"  # Needle pointing RIGHT
        syringe_start_col = np.min(x_coords)
        syringe_end_col = np.max(x_coords)
    else:
        orientation = "L"  # Needle pointing LEFT
        syringe_start_col = np.max(x_coords)
        syringe_end_col = np.min(x_coords)
    print(f"Syringe start col: {syringe_start_col}, end col: {syringe_end_col}")
    return image_rgb, orientation, mask, syringe_start_col, syringe_end_col


def detect_plunger(img, mask):
    """
    Detect the position of the black plunger in the segmented syringe image.

    Args:
        img (np.ndarray): RGB image
        mask (np.ndarray): Binary segmentation mask

    Returns:
        tuple: (
            rubber_mask (np.ndarray): Binary mask of detected plunger,
            start_col (int): Starting column of plunger window,
            end_col (int): Ending column of plunger window
        )
    """
    # Convert to grayscale and isolate syringe region
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply mask to grayscale image
    syringe_region = np.where(mask, gray, 255)  # Set non-syringe to white (255)

    # Visualize
    # plt.imshow(syringe_region, cmap="gray")
    # plt.title("Syringe Region Only (Grayscale)")
    # plt.axis("off")
    # plt.show()
    #
    # # Step: Find darkest areas (black rubber)
    # min_val = np.min(syringe_region)
    # black_threshold = min_val + 10  # Allow a small margin
    #
    # # Create mask of black rubber
    # rubber_mask = syringe_region < black_threshold
    #
    # # Visualize result
    # plt.imshow(rubber_mask, cmap='gray')
    # plt.title("Detected Black Rubber in Syringe")
    # plt.axis('off')
    # plt.show()
    # Step 1: Contrast Stretching
    syringe_enhanced = (
        syringe_region  # cv2.equalizeHist(syringe_region.astype(np.uint8))
    )

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(syringe_region, kernel, iterations=1)
    blurred = cv2.GaussianBlur(eroded, (3, 3), 0)

    # plt.imshow(blurred, cmap='gray')
    # plt.title("Syringe Region Only (Grayscale)")
    # plt.axis('off')
    # plt.show()

    # Threshold to find dark regions (plunger)
    _, rubber_mask = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    rubber_mask = cv2.morphologyEx(rubber_mask, cv2.MORPH_OPEN, kernel)

    # plt.imshow(rubber_mask, cmap="gray")
    # plt.title("Syringe Region Only (Grayscale)")
    # plt.axis("off")
    # plt.show()

    # Find optimal plunger window using sliding window
    height, width = rubber_mask.shape
    window_size = 20
    start_limit = int(0.25 * width)
    end_limit = int(0.75 * width)

    max_ratio = -1
    best_start = -1

    for col in range(start_limit, end_limit - window_size + 1):
        mask_window = mask[:, col : col + window_size]
        image_window = rubber_mask[:, col : col + window_size]
        # valid_pixels = image_window[mask_window]
        valid_pixels = image_window[mask_window > 0]

        if valid_pixels.size == 0:
            continue

        ratio = np.mean(valid_pixels)

        if ratio > max_ratio:
            max_ratio = ratio
            best_start = col
            # plt.imshow(rubber_mask, cmap='gray')
            # plt.axvline(best_start, color="yellow", linestyle="--", label="Syringe Start")
            # plt.axis('off')
            # plt.show()

    start_col = best_start
    end_col = best_start + window_size

    # Visualize plunger window on rubber mask
    print(f"Plunger window: {start_col}-{end_col}")

    return rubber_mask, start_col, end_col


def get_cut(
    flange_position,
    needletip_position,
    plunger_start,
    plunger_end,
    orientation,
    errorR,
    errorL,
):
    """
    Calculate the number of steps needed to move the paddle for cutting.

    Args:
        flange_position (float): Current flange position
        plunger_start (int): Starting position of plunger window
        plunger_end (int): Ending position of plunger window
        orientation (str): Syringe orientation ('L' or 'R')
        errorR (int): Error margin in pixels for the right oriented syringe
        errorL (int): Error margin in pixels for the left oriented syringe

    Returns:
        tuple: (
            steps (int): Number of steps to move the paddle,
            where_to_move (float): Position to move paddle to in inches,
            where_to_cut (float): Position to cut at in inches
        )
    """
    # Load calibration data
    _, _, _, _, _, _, pixel_to_length_ratio, _, BLADE_POS_LEFT, BLADE_POS_RIGHT = (
        undistort.load_calibration_data()
    )
    BLADE_POS_LEFT = BLADE_POS_LEFT / pixel_to_length_ratio
    BLADE_POS_RIGHT = BLADE_POS_RIGHT / pixel_to_length_ratio

    if orientation == "L":
        # Syringe pointed left - use right side of plunger window
        where_to_cut = (plunger_end) / pixel_to_length_ratio + errorL

        # If distance btw where to cut and flange is too long for distance btw two blades, discard syringe
        if (where_to_cut - needletip_position / pixel_to_length_ratio) > MAX_NEEDLE_LEN:
            print("Syringe too long!")
            where_to_move = MAX_PADDLE_POS
        elif where_to_cut < BLADE_POS_LEFT:
            where_to_move = 0
        else:
            where_to_move = MAX_PADDLE_POS - (
                BLADE_POS_LEFT
                - where_to_cut
                + flange_position / pixel_to_length_ratio
                + 0.2636
            )

    elif orientation == "R":
        # Syringe pointed right - use left side of plunger window
        where_to_cut = (plunger_end) / pixel_to_length_ratio - errorR
        if (needletip_position / pixel_to_length_ratio - where_to_cut) > MAX_NEEDLE_LEN:
            print("Syringe too long!")
            where_to_move = MAX_PADDLE_POS
        elif where_to_cut > BLADE_POS_RIGHT:
            where_to_move = 0
        else:
            where_to_move = (
                flange_position / pixel_to_length_ratio
                - where_to_cut
                + BLADE_POS_RIGHT
                - PADDLE_THICKNESS
                + 0.2636
            )
            print(f"Blade position right: {BLADE_POS_RIGHT}")

    elif orientation is None:
        # No syringe was detected
        where_to_move = 0

    else:
        raise ValueError("Invalid syringe orientation")

    return int(where_to_move * STEPS_TO_LENGTH_RATIO), where_to_move, where_to_cut


# Example usage
#
# # Open camera
# camera = cv2.VideoCapture(0)
#
# # Warm up camera
# for _ in range(5):
#     ret, frame = camera.read()
#
# if ret:
#     # Process the captured frame
#     image_rgb, orientation, mask, syringe_start_col, syringe_end_col = segment(frame)
#     if orientation != None:
#         rubber_mask, start_col, end_col = detect_plunger(image_rgb, mask)
#
#         # calculate cut
#         errorR = 0.8
#         errorL = 0.8
#         cut_steps, where_to_move, where_to_cut = get_cut(
#             syringe_start_col,
#             syringe_end_col,
#             start_col,
#             end_col,
#             orientation,
#             errorR,
#             errorL,
#         )
#
#         msg = f"<d{orientation}c{cut_steps}>"
#         print(f"sent {msg} to arduino")
#
#         # set up plot
#         fig, ax = plt.subplots()
#         image_disp = ax.imshow(image_rgb)
#         mask_disp = ax.imshow(mask, alpha=0.5)
#
#         # blades
#         (
#             _,
#             _,
#             _,
#             _,
#             _,
#             bounding_box,
#             pixel_to_length_ratio,
#             _,
#             BLADE_POS_LEFT,
#             BLADE_POS_RIGHT,
#         ) = undistort.load_calibration_data()
#         # Adjust blade positions for cropped image coordinates
#         x1 = int(bounding_box[0])
#         blade_line_L = ax.axvline(
#             BLADE_POS_LEFT,
#             color="red",
#             linestyle="--",
#             label="Blade L",
#         )
#         blade_line_R = ax.axvline(
#             BLADE_POS_RIGHT,
#             color="red",
#             linestyle="--",
#             label="Blade R",
#         )
#
#         where_to_move_line = ax.axvline(
#             int(where_to_move * pixel_to_length_ratio),
#             color="green",
#             linestyle="--",
#             label="where_to_move",
#         )
#         where_to_cut_line = ax.axvline(
#             int(where_to_cut
#             * pixel_to_length_ratio),  # Convert to pixels and adjust for cropped coordinates
#             color="blue",
#             linestyle="--",
#             label="where_to_cut",
#         )
#
#         ax.set_title("Predicted Mask from SAM")
#         ax.axis("off")
#         # add orientation text below image
#         orientation_text = fig.text(0.5, 0.05, orientation, ha="center", fontsize=12)
#
#         # update data in existing plot objects
#         mask_disp.set_alpha(0.5)
#         mask_disp.set_cmap("gray")
#
#         plt.legend()
#         plt.show()
#
#         # Visualize final results
#         plt.figure(figsize=(15, 5))
#
#         # Original with mask overlay
#         plt.subplot(131)
#         plt.imshow(image_rgb)
#         plt.imshow(mask, alpha=0.3)
#         plt.axvline(
#             syringe_start_col, color="yellow", linestyle="--", label="Syringe Start"
#         )
#         plt.axvline(
#             syringe_end_col, color="purple", linestyle="--", label="Syringe End"
#         )
#         plt.title(f"Segmentation Result\nOrientation: {orientation}")
#         plt.axis("off")
#         plt.legend()
#
#         # Plunger detection
#         plt.subplot(132)
#         plt.imshow(rubber_mask, cmap="gray")
#         plt.axvline((start_col+end_col)/2, color="lime", linestyle="--", label="Plunger")
#         # plt.axvline(end_col, color="orange", linestyle="--", label="Plunger End")
#         plt.title("Plunger Detection")
#         plt.legend()
#         plt.axis("off")
#
#         # Combined view
#         plt.subplot(133)
#         plt.imshow(image_rgb)
#         plt.imshow(rubber_mask, alpha=0.5, cmap="Reds")
#         plt.title("Combined View")
#         plt.axis("off")
#
#         plt.tight_layout()
#         plt.show()
#
#         # Save the captured frame if needed
#         cv2.imwrite("captured_frame.jpg", frame)
#
#         # Clean up
#         camera.release()
#     else:
#         cv2.imshow("No syringe present!", frame)
#         cv2.waitKey(0)  # press 0 to close
#         cv2.destroyAllWindows()
#
# else:
#     print("Failed to capture image from camera")
#
