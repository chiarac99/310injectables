import tkinter as tk
from tkinter import scrolledtext
import threading
import serial
import time

# === GUI + Serial Manager ===
class SerialGUI:
    def __init__(self, master, on_snap_callback):
        self.master = master
        master.title("Test Controller")
        master.geometry("400x300")

        self.on_snap_callback = on_snap_callback
        self.serial_port_name = '/dev/cu.usbmodem1201'
        self.serial_port = None
        self.running = False
        self.snap_received = False

        # Connect button
        self.connect_button = tk.Button(
            master, text="Connect", command=self.connect_serial
        )
        self.connect_button.pack(pady=5)

        # Control buttons
        self.ready_button = tk.Button(
            master, text="Send READY",
            command=lambda: self.send_command("READY"),
            state="disabled"
        )
        self.ready_button.pack(pady=5)

        self.pause_button = tk.Button(
            master, text="Send PAUSE",
            command=lambda: self.send_command("PAUSE"),
            state="disabled"
        )
        self.pause_button.pack(pady=5)

        self.resume_button = tk.Button(
            master, text="Send RESUME",
            command=lambda: self.send_command("RESUME"),
            state="disabled"
        )
        self.resume_button.pack(pady=5)

        self.output = scrolledtext.ScrolledText(master, width=60, height=15)
        self.output.pack(pady=10)

    def connect_serial(self):
        try:
            self.serial_port = serial.Serial(self.serial_port_name, 115200, timeout=1)
            time.sleep(2)  # Allow Arduino reset
            self.output.insert(tk.END, f"Connected to {self.serial_port_name}\n")
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
        run_main_test_logic(gui.serial_port_name)

    root = tk.Tk()
    gui = SerialGUI(root, on_snap_callback=on_snap)
    root.protocol("WM_DELETE_WINDOW", gui.on_close)
    root.mainloop()

# === Original test.py logic ===
def run_main_test_logic(arduino_port):
    import cv2
    import segment
    import undistort

    baud_rate = 115200
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)

    camera = cv2.VideoCapture(0)
    for _ in range(5):
        ret, _ = camera.read()

    try:
        while True:
            print("Waiting for SNAP command...")
            while True:
                try:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line == "SNAP":
                        print("Received SNAP from Arduino.")
                        break
                except Exception as e:
                    print(f"Ignoring serial read error: {e}")
                    continue

            ret, latest_frame = camera.read()
            if not ret:
                print("Error: Could not capture frame")
                continue

            image_rgb, orientation, mask, syringe_start_col, syringe_end_col = (
                segment.segment(latest_frame)
            )

            if orientation is None:
                print("No syringe detected")
                msg = "<dNc0>"
                print(f"sent {msg} to arduino")
                ser.write(msg.encode("utf-8"))
                continue

            _, plunger_start_col, plunger_end_col = segment.detect_plunger(image_rgb, mask)

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

            print(f"Orientation: {orientation}")
            print(f"Cut steps: {cut_steps}")
            print(f"Where to move: {where_to_move:.2f} inches")
            print(f"Where to cut: {where_to_cut:.2f} inches")
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nExiting program...")
    finally:
        camera.release()
        ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_gui_and_wait()
