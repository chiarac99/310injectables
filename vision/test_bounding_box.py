"""
Test script to verify if the bounding box is correctly cropping the image.
Shows original image with bounding box overlay and the cropped result.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from undistort import load_calibration_data


def test_bounding_box():
    # Open camera
    camera = cv2.VideoCapture(0)

    # Warm up camera
    for _ in range(5):
        ret, frame = camera.read()

    if not ret:
        print("Failed to capture image!")
        return

    # Get current bounding box and convert to integers
    _, _, _, _, _, bounding_box, _, _, bladeL_pixels_x, bladeR_pixels_x = (
        load_calibration_data()
    )
    print(f"\nRaw bounding box data: {bounding_box}")

    # Handle both (x,y,w,h) and (x1,y1,x2,y2) formats
    if len(bounding_box) == 3:
        x, y, w = bounding_box
        h = w  # Assume square if height is missing
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + w)
    else:
        x1, y1, x2, y2 = [int(coord) for coord in bounding_box]

    # Print the values for verification
    print(f"\nBounding Box Coordinates:")
    print(f"x1, y1: ({x1}, {y1})")
    print(f"x2, y2: ({x2}, {y2})")
    print(f"Width: {x2 - x1}, Height: {y2 - y1}")
    print(f"Original image shape: {frame.shape}\n")
    print(f"Blade L position: {bladeL_pixels_x}")
    print(f"Blade R position: {bladeR_pixels_x}")

    # Ensure coordinates are within image bounds
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Convert blade positions to integers
    bladeL_x = int(bladeL_pixels_x)
    bladeR_x = int(bladeR_pixels_x)

    # Create a copy of the frame to draw on
    frame_with_box = frame.copy()

    # Draw the bounding box on the original image
    cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw blade positions
    cv2.line(
        frame_with_box, (bladeL_x, y1), (bladeL_x, y2), (0, 0, 255), 2
    )  # Red line for left blade
    cv2.line(
        frame_with_box, (bladeR_x, y1), (bladeR_x, y2), (255, 0, 0), 2
    )  # Blue line for right blade

    try:
        # Crop the image using the bounding box
        cropped = frame[y1:y2, x1:x2]
        print(f"Cropped image shape: {cropped.shape}")

        # Convert from BGR to RGB for matplotlib
        frame_with_box_rgb = cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Create figure with two subplots
        plt.figure(figsize=(15, 5))

        # Original image with bounding box
        plt.subplot(121)
        plt.imshow(frame_with_box_rgb)
        plt.title(
            f"Original with Bounding Box and Blades\nBox coords: ({x1}, {y1}, {x2}, {y2})"
        )
        plt.axis("on")  # Keep axis on to see coordinates

        # Cropped image
        plt.subplot(122)
        plt.imshow(cropped_rgb)
        plt.title(f"Cropped Image\nShape: {cropped.shape}")
        plt.axis("on")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during cropping: {e}")
        print(
            "This might indicate that the bounding box coordinates are outside the image dimensions."
        )

    # Clean up
    camera.release()


if __name__ == "__main__":
    test_bounding_box()
