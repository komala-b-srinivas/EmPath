import cv2
import os

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"

subject_name = "102214_w_36"
sample_name  = "102214_w_36-PA1-014"

video_path = os.path.join(BIOVID_ROOT, "video", subject_name, sample_name + ".mp4")

print(f"Looking for : {video_path}")
print(f"File exists : {os.path.exists(video_path)}")

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Total frames : {total_frames}")
print(f"FPS          : {fps}")
print(f"Resolution   : {width} x {height}")

ret, frame = cap.read()
if ret:
    cv2.imwrite("test_frame.jpg", frame)
    print("Saved test_frame.jpg — open it to confirm you see a face")
else:
    print("ERROR: Could not read frame")

cap.release()