import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import serial
import serial.tools.list_ports
import time
import sys


# === GUI + Serial Manager ===
class SerialGUI:
    def __init__(self, master, on_snap_callback):
        self.master = master
        master.title("Test Controller")

        self.on_snap_callback = on_snap_callback
        self.serial_port = None
        self.running = False
        self.paused = False
        self.snap_received = False

        # Port selection
        self.port_var = tk.StringVar()
        self.port_menu = ttk.Combobox(
            master, textvariable=self.port_var, state="readonly"
        )
        self.port_menu["values"] = self.get_serial_ports()
        self.port_menu.pack()
        self.port_menu.set("Select Serial Port")

        self.connect_button = tk.Button(
            master, text="Connect", command=self.connect_serial
        )
        self.connect_button.pack()

        # Control buttons
        self.ready_button = tk.Button(
            master,
            text="Send READY",
            command=lambda: self.send_command("READY"),
            state="disabled",
        )
        self.ready_button.pack()

        self.pause_button = tk.Button(
            master,
            text="Send PAUSE",
            command=lambda: self.send_command("PAUSE"),
            state="disabled",
        )
        self.pause_button.pack()

        self.resume_button = tk.Button(
            master,
            text="Send RESUME",
            command=lambda: self.send_command("RESUME"),
            state="disabled",
        )
        self.resume_button.pack()

        self.output = scrolledtext.ScrolledText(master, width=60, height=15)
        self.output.pack()

    def get_serial_ports(self):
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect_serial(self):
        port = self.port_var.get()
        if port and port != "Select Serial Port":
            try:
                self.serial_port = serial.Serial(port, 115200, timeout=1)
                time.sleep(2)
                self.output.insert(tk.END, f"Connected to {port}\n")
                self.running = True
                self.ready_button.config(state="normal")
                self.pause_button.config(state="normal")
                self.resume_button.config(state="normal")
                self.thread = threading.Thread(target=self.read_serial, daemon=True)
                self.thread.start()
            except serial.SerialException as e:
                self.output.insert(tk.END, f"Failed to connect: {e}\n")

    def send_command(self, cmd):
        if self.serial_port:
            self.serial_port.write((cmd + "\n").encode())
            self.output.insert(tk.END, f">> Sent: {cmd}\n")

    def read_serial(self):
        while self.running:
            if self.serial_port and self.serial_port.in_waiting:
                line = self.serial_port.readline().decode().strip()
                if line:
                    self.output.insert(tk.END, f"<< {line}\n")
                    self.output.see(tk.END)
                    if line == "SNAP" and not self.snap_received:
                        self.snap_received = True
                        self.on_snap_callback()
            time.sleep(0.1)

    def on_close(self):
        self.running = False
        if self.serial_port:
            self.serial_port.close()
        self.master.destroy()


# === Start GUI and wait for SNAP ===
def start_gui_and_wait():
    def on_snap():
        if not gui.serial_port:
            gui.output.insert(tk.END, "Error: No serial port connected\n")
            return
        gui.output.insert(tk.END, "!! SNAP received - running test logic...\n")
        gui.output.see(tk.END)
        run_main_test_logic(gui.port_var.get())

    root = tk.Tk()
    gui = SerialGUI(root, on_snap_callback=on_snap)
    root.protocol("WM_DELETE_WINDOW", gui.on_close)
    root.mainloop()


# === Original test.py logic embedded ===
def run_main_test_logic(arduino_port):
    import cv2
    import segment
    import undistort
    import serial
    import time

    # serial
    baud_rate = 115200
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)  # wait for Arduino to reset

    # Open camera
    camera = cv2.VideoCapture(0)
    # warm up camera silently
    for _ in range(5):
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

            # take new image silently
            ret, latest_frame = camera.read()
            if not ret:
                print("Error: Could not capture frame")
                continue

            # get orientation and syringe mask
            image_rgb, orientation, mask, syringe_start_col, syringe_end_col = (
                segment.segment(latest_frame)
            )

            if orientation is None:
                print("No syringe detected")
                msg = "<dNc0>"  # Special message for no detection
                print(f"sent {msg} to arduino")
                ser.write(msg.encode("utf-8"))
                continue

            # get plunger position
            _, plunger_start_col, plunger_end_col = segment.detect_plunger(
                image_rgb, mask
            )

            # calculate cut
            errorR = 0.9
            errorL = 0.8
            cut_steps, where_to_move, where_to_cut = segment.get_cut(
                syringe_start_col,
                syringe_end_col,
                plunger_start_col,
                plunger_end_col,
                orientation,
                errorR,
                errorL,
            )

            msg = f"<d{orientation}c{cut_steps}>"
            print(f"sent {msg} to arduino")
            ser.write(msg.encode("utf-8"))

            # Display basic info
            print(f"Orientation: {orientation}")
            print(f"Cut steps: {cut_steps}")
            print(f"Where to move: {where_to_move:.2f} inches")
            print(f"Where to cut: {where_to_cut:.2f} inches")
            print("-" * 50)

            # # Visualization code (uncomment to use)
            # # set up plot
            # fig, ax = plt.subplots()
            # image_disp = ax.imshow(image_rgb)
            # mask_disp = ax.imshow(mask, alpha=0.5)
            #
            # # blades
            # _, _, _, _, _, _, pixels_to_length_ratio = undistort.load_calibration_data()
            # blade_line_L = ax.axvline(
            #     segment.BLADE_POS_LEFT * pixels_to_length_ratio,
            #     color="red",
            #     linestyle="--",
            #     label="Blade L",
            # )
            # blade_line_R = ax.axvline(
            #     segment.BLADE_POS_RIGHT * pixels_to_length_ratio,
            #     color="red",
            #     linestyle="--",
            #     label="Blade R",
            # )
            #
            # where_to_move_line = ax.axvline(
            #     where_to_move * pixels_to_length_ratio,
            #     color="green",
            #     linestyle="--",
            #     label="where_to_move",
            # )
            # where_to_cut_line = ax.axvline(
            #     where_to_cut * pixels_to_length_ratio,
            #     color="blue",
            #     linestyle="--",
            #     label="where_to_cut",
            # )
            #
            # ax.set_title("Predicted Mask from SAM")
            # ax.axis("off")
            # # add orientation text below image
            # orientation_text = fig.text(0.5, 0.05, orientation, ha="center", fontsize=12)
            #
            # # update data in existing plot objects
            # mask_disp.set_alpha(0.5)
            # mask_disp.set_cmap("gray")
            #
            # plt.legend()
            # plt.show()

    except KeyboardInterrupt:
        print("\nExiting program...")
    finally:
        # Clean up
        camera.release()
        ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_gui_and_wait()
