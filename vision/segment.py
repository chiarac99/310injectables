"""
segment.py

Methods for segmentation of  a given image and determining position of the black plunger.

"""

import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import undistort
from vision.undistort import undistort_img


def segment(img):
    """

    Args:
        img:

    Returns:

    """

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sam_checkpoint = "vision/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # get bounding box
    _, _, _, _, _, bounding_box = undistort.load_calibration_data()

    (x1, y1, x2, y2) = bounding_box
    image_rgb = image_rgb[y1:y2, x1:x2]
    # cv2.imshow("image_rgb", image_rgb)
    # cv2.waitKey(0)  # press 0 to close
    # cv2.destroyAllWindows()

    predictor.set_image(image_rgb)

    x, y, z = np.shape(image_rgb)
    input_box = np.array([0, 0, y, x])  # Just the shape of the image
    input_label = np.array([0])

    masks, scores, logits = predictor.predict(
        box=input_box[None, :],
        multimask_output=False
    )

    mask = masks[0]  # single mask from box
    mask = ~mask  # Sometimes, segmentation takes the background as foreground. I am still looking to correct it automatically
    # plt.figure()
    # plt.imshow(image_rgb)
    # plt.imshow(mask, alpha=0.5)
    # plt.title("Step 5: Predicted Mask from SAM")
    # plt.axis('off')
    # plt.show()

    column_sums = np.sum(mask, axis=0)
    max_width_col = np.argmax(column_sums)
    max_width_value = column_sums[max_width_col]

    print("Max width column:", max_width_col)
    print("Max width value (pixels):", max_width_value)

    # plt.imshow(mask, cmap='gray')
    # plt.axvline(max_width_col, color='red', linestyle='--')
    # plt.title(f"Max Width Col: {max_width_col}")
    # plt.show()

    x_foreground = np.any(mask, axis=0)  # Boolean array: columns with any mask
    x_coords = np.where(x_foreground)[0]  # Column indices with mask
    x_center = int(np.mean(x_coords))  # Mean column index

    if max_width_col < x_center:
        orientation = "Needle is pointing RIGHT"
    else:
        orientation = "Needle is pointing LEFT"

    print("🟢", orientation)

    return image_rgb, orientation, mask


def detect_plunger(img, mask):
    """

    Args:
        img:

    Returns:

    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply mask to grayscale image
    syringe_region = np.where(mask, gray, 255)  # Set non-syringe to white (255)

    # # Visualize
    # plt.imshow(syringe_region, cmap='gray')
    # plt.title("Syringe Region Only (Grayscale)")
    # plt.axis('off')
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
    syringe_enhanced = syringe_region #cv2.equalizeHist(syringe_region.astype(np.uint8))

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(syringe_enhanced, kernel, iterations=1)

    # Step 2: Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(eroded, (3, 3), 0)

    # Step 3: Erosion to remove small bright noise

    # Step 4: Threshold to find darkest regions
    _, rubber_mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Step 5: Morphological cleanup (remove small holes)
    rubber_mask = cv2.morphologyEx(rubber_mask, cv2.MORPH_OPEN, kernel)

    # # Step 6: Show the result
    # plt.imshow(rubber_mask, cmap='gray')
    # plt.title("Rubber Detection with Blurring + Erosion + Threshold + Morphology")
    # plt.axis('off')
    # plt.show()

    # Image size
    height, width = rubber_mask.shape

    # Sliding window config
    window_size = 20
    start_limit = int(0.25 * width)
    end_limit = int(0.75 * width)

    # Tracking best window
    max_ratio = -1
    best_start = -1

    for col in range(start_limit, end_limit - window_size + 1):
        # Define the current window slice
        mask_window = mask[:, col:col + window_size]
        image_window = rubber_mask[:, col:col + window_size]

        # Extract only pixels within the mask
        valid_pixels = image_window[mask_window]

        if valid_pixels.size == 0:
            continue  # skip if no mask present

        ratio = np.mean(valid_pixels)  # or sum(valid_pixels)/count — same here
        if ratio > max_ratio:
            max_ratio = ratio
            best_start = col

    # Result
    start_col = best_start
    end_col = best_start + window_size

    # print(f"Best region inside mask: {start_col}–{end_col}, avg intensity = {max_ratio:.2f}")

    # Visualization
    # plt.imshow(gray, cmap='gray')
    # plt.axvline(start_col, color='lime', linestyle='--', label='Window Start')
    # plt.axvline(end_col, color='orange', linestyle='--', label='Window End')
    # plt.title(f"Max Proportional Brightness in Mask | {start_col}-{end_col}")
    # plt.legend()
    # plt.axis('off')
    # plt.show()

    # # Image size
    # height, width = rubber_mask.shape
    #
    # # Sliding window config
    # window_size = 20
    # start_limit = int(0.25 * width)
    # end_limit = int(0.75 * width)
    #
    # # Tracking best window
    # max_ratio = -1
    # best_start = -1
    #
    # for col in range(start_limit, end_limit - window_size + 1):
    #     # Define the current window slice
    #     mask_window = mask[:, col:col + window_size]
    #     image_window = rubber_mask[:, col:col + window_size]
    #
    #     # Extract only pixels within the mask
    #     valid_pixels = image_window[mask_window]
    #
    #     if valid_pixels.size == 0:
    #         continue  # skip if no mask present
    #
    #     ratio = np.mean(valid_pixels)  # or sum(valid_pixels)/count — same here
    #     if ratio > max_ratio:
    #         max_ratio = ratio
    #         best_start = col
    #
    # # Result
    # start_col = best_start
    # end_col = best_start + window_size
    #
    # print(f"Best region inside mask: {start_col}–{end_col}, avg intensity = {max_ratio:.2f}")
    #
    # # Visualization
    # plt.imshow(rubber_mask, cmap='gray')
    # plt.axvline(start_col, color='lime', linestyle='--', label='Window Start')
    # plt.axvline(end_col, color='orange', linestyle='--', label='Window End')
    # plt.title(f"Max Proportional Brightness in Mask | {start_col}-{end_col}")
    # plt.legend()
    # plt.axis('off')
    # plt.show()

    return rubber_mask, start_col, end_col


# image = cv2.imread("test/syringe_new.jpg")
#
# # Find bounding box here
# # Uncomment this code and draw a bounding box
# # this will give you x, y, width and height coordinates of the important step
# # r = cv2.selectROI("Select Bounding Box", image, fromCenter=False, showCrosshair=True)
# # print("Bounding box:", r)  # (x, y, width, height)
# # cv2.destroyAllWindows()
#
# image_rgb, orientation, mask = segment(image)
#
# detect_plunger(image_rgb, mask)