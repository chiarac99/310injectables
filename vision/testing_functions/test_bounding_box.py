"""
Test script to verify if the bounding box is correctly cropping the image.
Shows original image with bounding box overlay and the cropped result.
"""

import cv2
import matplotlib.pyplot as plt
from vision.undistort import load_calibration_data


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

    # Calculate box width and expected blade positions
    box_width = x2 - x1
    expected_bladeL = int(box_width * (3.5 / 11))
    expected_bladeR = int(box_width * (7.5 / 11))

    # Print the values for verification
    print(f"\nBounding Box Coordinates:")
    print(f"x1, y1: ({x1}, {y1})")
    print(f"x2, y2: ({x2}, {y2})")
    print(f"Box width: {box_width}")
    print(f"Original image shape: {frame.shape}\n")
    print(f"Expected blade positions (relative to box):")
    print(f"Left blade: {expected_bladeL} pixels ({(3.5/11)*100:.1f}% of box width)")
    print(f"Right blade: {expected_bladeR} pixels ({(7.5/11)*100:.1f}% of box width)")
    print(f"Actual blade positions (from calibration):")
    print(
        f"Left blade: {bladeL_pixels_x} pixels ({(bladeL_pixels_x/box_width)*100:.1f}% of box width)"
    )
    print(
        f"Right blade: {bladeR_pixels_x} pixels ({(bladeR_pixels_x/box_width)*100:.1f}% of box width)"
    )

    # Ensure coordinates are within image bounds
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Create a copy of the frame to draw on
    frame_with_box = frame.copy()

    # Draw the bounding box on the original image
    cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

    try:
        # Crop the image using the bounding box
        cropped = frame[y1:y2, x1:x2]
        print(f"Cropped image shape: {cropped.shape}")

        # Create a copy of cropped image to draw blade lines
        cropped_with_blades = cropped.copy()

        # Draw vertical lines at 10% intervals for reference
        for i in range(1, 10):
            x_pos = int(box_width * (i / 10))
            cv2.line(
                cropped_with_blades,
                (x_pos, 0),
                (x_pos, cropped.shape[0]),
                (128, 128, 128),
                1,
            )  # Gray lines at 10% intervals
            # Add percentage labels at the top
            cv2.putText(
                cropped_with_blades,
                f"{i*10}%",
                (x_pos - 20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (128, 128, 128),
                1,
            )

        # Draw blade positions on cropped image
        # Left blade (red)
        cv2.line(
            cropped_with_blades,
            (expected_bladeL, 0),
            (expected_bladeL, cropped.shape[0]),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            cropped_with_blades,
            "L",
            (expected_bladeL - 10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Right blade (blue)
        cv2.line(
            cropped_with_blades,
            (expected_bladeR, 0),
            (expected_bladeR, cropped.shape[0]),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            cropped_with_blades,
            "R",
            (expected_bladeR - 10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # Convert from BGR to RGB for matplotlib
        frame_with_box_rgb = cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB)
        cropped_with_blades_rgb = cv2.cvtColor(cropped_with_blades, cv2.COLOR_BGR2RGB)

        # Create figure with two subplots
        plt.figure(figsize=(15, 5))

        # Original image with bounding box
        plt.subplot(121)
        plt.imshow(frame_with_box_rgb)
        plt.title(f"Original with Bounding Box\nBox coords: ({x1}, {y1}, {x2}, {y2})")
        plt.axis("on")  # Keep axis on to see coordinates

        # Cropped image with blade positions
        plt.subplot(122)
        plt.imshow(cropped_with_blades_rgb)
        plt.title(
            f"Cropped Image with Blade Positions\n"
            + f"Left Blade at {(expected_bladeL/box_width)*100:.1f}%, Right Blade at {(expected_bladeR/box_width)*100:.1f}%"
        )
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
