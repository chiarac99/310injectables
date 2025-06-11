import cv2
import time


def test_camera():
    # Initialize camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera opened successfully")

    # Warm up camera
    print("Warming up camera...")
    for i in range(5):
        ret, _ = cap.read()
        if not ret:
            print(f"Failed to read frame during warmup {i+1}")
        time.sleep(0.1)

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Print frame info
            print(f"Frame shape: {frame.shape}")
            print(f"Frame dtype: {frame.dtype}")
            print(f"Frame min/max values: {frame.min()}, {frame.max()}")

            # Display frame
            cv2.imshow("Camera Test", frame)

            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == "__main__":
    test_camera()
