import cv2
import segment
import undistort
import matplotlib.pyplot as plt
import serial
import time

# serial
arduino_port = "/dev/cu.usbmodem14101"
baud_rate = 115200
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # wait for Arduino to reset

# Open camera
camera = cv2.VideoCapture(0)
# warm up camera
for _ in range(5):
    # take image
    ret, _ = camera.read()

try:
    while True:
        print("Waiting for SNAP command...")
        # Wait for SNAP, ignoring any invalid data
        while True:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line == "SNAP":
                    print("Received SNAP from Arduino.")
                    break
            except Exception as e:
                print(f"Ignoring serial read error: {e}")
                continue

        # take new image
        ret, latest_frame = camera.read()
        undistorted_frame = (
            latest_frame  # undistort.undistort_img(latest_frame, display=False)
        )

        # get orientation and syringe mask
        image_rgb, orientation, mask, syringe_start_col, syringe_end_col = segment.segment(
            undistorted_frame
        )

        # get plunger position
        _, plunger_start_col, plunger_end_col = segment.detect_plunger(image_rgb, mask)

        # calculate cut
        error = 30
        cut_steps, where_to_move, where_to_cut = segment.get_cut(
            syringe_start_col,
            syringe_end_col,
            plunger_start_col,
            plunger_end_col,
            orientation,
            error,
        )

        msg = f"<d{orientation}c{cut_steps}>"
        print(f"sent {msg} to arduino")
        ser.write(msg.encode("utf-8"))

        # set up plot
        fig, ax = plt.subplots()
        image_disp = ax.imshow(image_rgb)
        mask_disp = ax.imshow(mask, alpha=0.5)

        # blades
        _, _, _, _, _, _, pixels_to_length_ratio = undistort.load_calibration_data()
        blade_line_L = ax.axvline(
            segment.BLADE_POS_LEFT * pixels_to_length_ratio,
            color="red",
            linestyle="--",
            label="Blade L",
        )
        blade_line_R = ax.axvline(
            segment.BLADE_POS_RIGHT * pixels_to_length_ratio,
            color="red",
            linestyle="--",
            label="Blade R",
        )

        where_to_move_line = ax.axvline(
            where_to_move * pixels_to_length_ratio,
            color="green",
            linestyle="--",
            label="where_to_move",
        )
        where_to_cut_line = ax.axvline(
            where_to_cut * pixels_to_length_ratio,
            color="blue",
            linestyle="--",
            label="where_to_cut",
        )

        ax.set_title("Predicted Mask from SAM")
        ax.axis("off")
        # add orientation text below image
        orientation_text = fig.text(0.5, 0.05, orientation, ha="center", fontsize=12)

        # update data in existing plot objects
        mask_disp.set_alpha(0.5)
        mask_disp.set_cmap("gray")

        plt.legend()
        plt.show()

except KeyboardInterrupt:
    print("\nExiting program...")
finally:
    # Clean up
    camera.release()
    ser.close()
    cv2.destroyAllWindows()
