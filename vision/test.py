import cv2
import segment
import undistort
import matplotlib.pyplot as plt

# # Open camera
camera = cv2.VideoCapture(0)
# warm up camera
for _ in range(5):
    # take image
    ret, latest_frame = camera.read()

undistorted_frame = latest_frame #undistort.undistort_img(latest_frame, display=False)

# latest_frame = cv2.imread("in_situ/photo_1747768586.jpg")

image_rgb, orientation, mask = segment.segment(latest_frame)

rubber_mask, start_col, end_col = segment.detect_plunger(image_rgb, mask)

print(start_col)
print(end_col)

fig0, ax0 = plt.subplots()
ax0.imshow(image_rgb)

# --- Step 4: Create first figure: image + mask overlay ---
fig1, ax1 = plt.subplots()
ax1.imshow(image_rgb)
ax1.imshow(mask, alpha=0.5)
ax1.set_title("Predicted Mask from SAM")
ax1.axis('off')

# --- Step 5: Create second figure: rubber mask + bounding lines ---
fig2, ax2 = plt.subplots()
ax2.imshow(rubber_mask, cmap='gray')
ax2.axvline(start_col, color='lime', linestyle='--', label='Window Start')
ax2.axvline(end_col, color='orange', linestyle='--', label='Window End')
ax2.set_title("Detected Black Rubber in Syringe")
ax2.axis('off')
ax2.legend(loc='lower center')

# --- Step 6: Show both figures ---
plt.show()