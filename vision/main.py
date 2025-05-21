import serial
import cv2
import time
import numpy as np
import os
import matplotlib.pyplot as plt

import segment
import undistort

arduino_port = '/dev/cu.usbmodem11301'
baud_rate = 115200

plunger_detection = True  # TODO: remove this, this is just in case during the demo Tues 20 May plunger detection not working

# create dir for photos if it does not already exist0
os.makedirs("in_situ", exist_ok=True)

# Open serial port
# ser = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # Wait for Arduino to initialize

# Open camera
camera = cv2.VideoCapture(0)

# initialise image_rgb as a black image and mask as blank too
image_rgb = np.zeros((138, 1311, 3), dtype=np.uint8)
mask = np.zeros((138, 1311), dtype=bool)  # 2D boolean array (same width/height as image)
orientation = "Waiting for orientation..."
plunger_start_col = 0

# Enable interactive mode
plt.ion()

# set up plot
fig, ax = plt.subplots()
image_disp = ax.imshow(image_rgb)
mask_disp = ax.imshow(mask, alpha=0.5)
plunger_line = ax.axvline(plunger_start_col, color='lime', linestyle='--', label='Window Start')
ax.set_title("Predicted Mask from SAM")
ax.axis('off')
# add orientation text below image
orientation_text = fig.text(0.5, 0.05, orientation, ha='center', fontsize=12)


while True:
    # show live preview of image

    # listen to serial while displaying image
    if True: #ser.in_waiting:
        #command = ser.readline().decode().strip()
        if True:#command == "SNAP":
            # command to take an image received

            # warm up camera
            for _ in range(5):
                # take image
                ret, latest_frame = camera.read()

            if ret:
                # valid image so update the frame
                frame = latest_frame

                # undistort image
                undistorted_frame = latest_frame #undistort.undistort_img(latest_frame, display=False)

                # get orientation and syringe mask
                image_rgb, orientation, mask = segment.segment(undistorted_frame)

                # get plunger position
                # TODO: send plunger position and orientation to arduino via serial
                _, plunger_start_col, plunger_end_col = segment.detect_plunger(image_rgb, mask)
                print(plunger_start_col)
                print(plunger_end_col)

                # save image
                filename = f"in_situ/photo_{int(time.time())}.jpg"
                cv2.imwrite(filename, undistorted_frame)
                print(f"saved {filename}")

                # update data in existing plot objects
                # image_disp.set_data(image_rgb)
                image_disp.remove()
                image_disp = ax.imshow(image_rgb)
                # mask_disp.set_data(mask)
                # Remove old mask from plot
                mask_disp.remove()
                # Add new mask (make sure to convert to a supported dtype)
                mask_disp = ax.imshow(mask.astype(float), alpha=0.5, cmap='gray')

                # update plunger line display
                # remove the old line
                plunger_line.remove()
                # create a new one
                plunger_line = ax.axvline(sum([plunger_start_col, plunger_end_col])/len([plunger_start_col, plunger_end_col]), color='lime', linestyle='--', label='Window Start')

                # update text
                # orientation_text.set_text(orientation)
                orientation_text.remove()
                orientation_text = fig.text(0.5, 0.05, orientation, ha='center', fontsize=12)

                # refresh the plot without blocking
                fig.canvas.draw()
                fig.canvas.flush_events()

                time.sleep(1)

            else:
                print("❌ Failed to read from camera")
