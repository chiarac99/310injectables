import serial
import cv2
import time
import numpy as np
import os
import matplotlib.pyplot as plt

import segment
import undistort
from vision.segment import syringe_end_col

# Mode control: True for single frame, False for continuous
single_frame = True

arduino_port = "/dev/cu.usbmodem11301"
baud_rate = 115200

plunger_detection = True  # TODO: remove this, this is just in case during the demo Tues 20 May plunger detection not working

# create dir for photos if it does not already exist
os.makedirs("in_situ", exist_ok=True)

# Open camera
camera = cv2.VideoCapture(0)

# initialise image_rgb as a black image and mask as blank too
image_rgb = np.zeros((138, 1311, 3), dtype=np.uint8)
mask = np.zeros(
    (138, 1311), dtype=bool
)  # 2D boolean array (same width/height as image)
orientation = "Waiting for orientation..."
plunger_start_col = 0

# Enable interactive mode
plt.ion()

# set up plot
fig, ax = plt.subplots()
image_disp = ax.imshow(image_rgb)
mask_disp = ax.imshow(mask, alpha=0.5)
plunger_line = ax.axvline(
    plunger_start_col, color="lime", linestyle="--", label="Window Start"
)
ax.set_title("Predicted Mask from SAM")
ax.axis("off")
# add orientation text below image
orientation_text = fig.text(0.5, 0.05, orientation, ha="center", fontsize=12)


def process_frame():
    """Process a single frame from the camera"""
    # warm up camera
    for _ in range(5):
        # take image
        ret, latest_frame = camera.read()

    if ret:
        # valid image so update the frame
        frame = latest_frame

        # undistort image
        undistorted_frame = (
            latest_frame  # undistort.undistort_img(latest_frame, display=False)
        )

        # get orientation and syringe mask
        image_rgb, orientation, mask, syringe_start_col = segment.segment(
            undistorted_frame
        )

        # get plunger position
        _, plunger_start_col, plunger_end_col = segment.detect_plunger(image_rgb, mask)

        # calculate cut
        errorR = 60
        errorL = 30
        cut_steps = segment.get_cut(
            syringe_start_col,
            syringe_end_col,
            plunger_start_col,
            plunger_end_col,
            orientation,
            errorR,
            errorL
        )

        # save image
        filename = f"in_situ/photo_{int(time.time())}.jpg"
        cv2.imwrite(filename, undistorted_frame)
        print(f"saved {filename}")

        # update data in existing plot objects
        image_disp.set_data(image_rgb)
        mask_disp.set_data(mask.astype(float))
        mask_disp.set_alpha(0.5)
        mask_disp.set_cmap("gray")

        # Update plunger line position
        plunger_pos = sum([plunger_start_col, plunger_end_col]) / len(
            [plunger_start_col, plunger_end_col]
        )
        plunger_line.set_data([plunger_pos, plunger_pos], [0, 1])

        # Update orientation text
        orientation_text.set_text(orientation)

        # refresh the plot without blocking
        fig.canvas.draw()
        fig.canvas.flush_events()

        return True
    else:
        print("❌ Failed to read from camera")
        return False


def main():
    """Main function to run the system"""
    print(f"\nRunning in {'single frame' if single_frame else 'continuous'} mode")

    try:
        if not single_frame:  # continuous mode
            while True:
                process_frame()
                time.sleep(1)
        else:  # single frame mode
            process_frame()
            # Keep the plot window open until manually closed
            plt.show(block=True)

    except KeyboardInterrupt:
        print("\nStopping gracefully...")
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        plt.close()


if __name__ == "__main__":
    main()
