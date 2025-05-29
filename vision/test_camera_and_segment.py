"""
Test Camera and Segment: Quick script to capture an image and run segmentation

This script:
1. Captures an image from the camera
2. Runs segmentation using the SAM model
3. Displays the results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment import segment, detect_plunger
from undistort import undistort_img, load_calibration_data


def capture_and_segment():
    # Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera resolution (adjust if needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        # Load calibration data
        mtx, dist, _, _, _, _, _, _, _, _ = load_calibration_data()

        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Display the live feed
            cv2.imshow("Press SPACE to capture, Q to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):  # Space key to capture
                # Undistort the image
                undistorted = undistort_img(frame)

                # Run segmentation
                image_rgb, orientation, mask, start_col, end_col = segment(undistorted)

                if orientation is None:
                    print("No syringe detected in the image")
                    continue

                # Run plunger detection
                rubber_mask, plunger_start, plunger_end = detect_plunger(
                    image_rgb, mask
                )

                # Display results
                plt.figure(figsize=(15, 5))

                plt.subplot(131)
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title("Original Image")
                plt.axis("off")

                plt.subplot(132)
                plt.imshow(image_rgb)
                plt.imshow(mask, alpha=0.5)
                plt.title(f"Segmentation (Orientation: {orientation})")
                plt.axis("off")

                plt.subplot(133)
                plt.imshow(rubber_mask, cmap="gray")
                plt.title("Plunger Detection")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

    finally:
        # Release camera
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_segment()
