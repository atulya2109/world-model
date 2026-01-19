import cv2
import numpy as np
import glob
import json
import os

DATA_PATH = "dataset_raw/images/*.jpg"
output_path = "mask_coords.json"

files = sorted(glob.glob(DATA_PATH))
if not files:
    print(f"No images found matching {DATA_PATH}")
    exit(1)

image_path = files[0]
print(f"Using image: {image_path}")

img = cv2.imread(image_path)
if img is None:
    print(f"Failed to load {image_path}")
    exit(1)

original = img.copy()
selected_rect = []
is_dragging = False


def on_mouse(event, x, y, flags, param):
    global selected_rect, is_dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        selected_rect = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False
        selected_rect.append((x, y))

        x1, y1 = selected_rect[0]
        x2, y2 = selected_rect[1]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Select Mask Area", original)

        print(f"Selected: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        temp = original.copy()
        x1, y1 = selected_rect[0]
        cv2.rectangle(temp, (x1, y1), (x, y), (0, 255, 0), 2)
        cv2.imshow("Select Mask Area", temp)


cv2.namedWindow("Select Mask Area")
cv2.setMouseCallback("Select Mask Area", on_mouse)

cv2.imshow("Select Mask Area", original)
print("Click and drag to select an area. Press 's' to save, 'r' to reset, 'q' to quit.")

while True:
    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        selected_rect = []
        original = img.copy()
        cv2.imshow("Select Mask Area", original)
        print("Reset. Select a new area.")

    elif key == ord("s") and len(selected_rect) == 2:
        x1, y1 = selected_rect[0]
        x2, y2 = selected_rect[1]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        coords = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        with open(output_path, "w") as f:
            json.dump(coords, f, indent=2)
        print(f"Saved mask coordinates to {output_path}")

        mask_preview = img.copy()
        cv2.rectangle(mask_preview, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.imwrite("mask_preview.png", mask_preview)
        print("Saved mask_preview.png")

cv2.destroyAllWindows()
print("Done.")
