import cv2
import os
import numpy as np

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"

# Test with a PA3 sample - highest pain level
subject_name = "102214_w_36"
sample_name  = "102214_w_36-PA3-058"

video_path = os.path.join(BIOVID_ROOT, "video", subject_name, sample_name + ".mp4")

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
duration     = total_frames / fps

print(f"Total frames : {total_frames}")
print(f"FPS          : {fps}")
print(f"Duration     : {duration:.2f} seconds")

print(f"\nWith 10 frames evenly sampled:")
indices = [int(i * total_frames / 10) for i in range(10)]
print(f"Frame indices : {indices}")
print(f"Timestamps    : {[round(i/fps, 2) for i in indices]} seconds")

print(f"\nWith 10 frames from SECOND HALF only:")
indices_second_half = [int(total_frames//2 + i * (total_frames//2) / 10) for i in range(10)]
print(f"Frame indices : {indices_second_half}")
print(f"Timestamps    : {[round(i/fps, 2) for i in indices_second_half]} seconds")

# Save one frame from each approach so you can visually compare
os.makedirs("timing_test", exist_ok=True)

# Frame from first half
cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
ret, frame = cap.read()
if ret:
    cv2.imwrite("timing_test/first_half_frame.jpg", frame)

# Frame from second half
cap.set(cv2.CAP_PROP_POS_FRAMES, indices_second_half[5])
ret, frame = cap.read()
if ret:
    cv2.imwrite("timing_test/second_half_frame.jpg", frame)

print("\nSaved comparison frames to timing_test/ folder")
print("Open both images and compare expressions")

cap.release()