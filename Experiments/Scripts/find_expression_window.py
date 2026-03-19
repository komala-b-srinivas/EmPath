import cv2
import numpy as np
import os

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"

def find_expression_frames(video_path, num_frames=5):
    """
    Finds frames where facial expression is most active
    by detecting regions of highest motion/change.
    Returns list of frames from the most expressive window.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Step 1 — Read all frames and convert to grayscale
    all_frames = []
    gray_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(gray_frames) < 2:
        return all_frames[:num_frames]

    # Step 2 — Compute frame difference scores
    # How much does each frame differ from the previous one?
    diff_scores = [0]  # first frame has no diff
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
        score = np.mean(diff)
        diff_scores.append(score)

    diff_scores = np.array(diff_scores)

    # Step 3 — Find the window of highest activity
    # Use a rolling average to find the most active region
    window_size = int(fps * 1.5)  # 1.5 second window
    best_start  = 0
    best_score  = 0

    for i in range(len(diff_scores) - window_size):
        window_score = np.mean(diff_scores[i:i + window_size])
        if window_score > best_score:
            best_score = window_score
            best_start = i

    best_end = min(best_start + window_size, len(all_frames))

    # Step 4 — Sample num_frames evenly from this window
    indices = [int(best_start + i * (best_end - best_start) / num_frames)
               for i in range(num_frames)]

    selected_frames = [all_frames[i] for i in indices if i < len(all_frames)]

    return selected_frames, best_start / fps, best_end / fps, diff_scores


# ── Test on one PA3 and one PA2 video ─────────────────────────────────────
def test_video(subject_name, sample_name, label):
    video_path = os.path.join(BIOVID_ROOT, "video", subject_name,
                              sample_name + ".mp4")

    frames, start_t, end_t, diff_scores = find_expression_frames(video_path)

    print(f"\n{label} — {sample_name}")
    print(f"  Most active window: {start_t:.2f}s → {end_t:.2f}s")
    print(f"  Max diff score    : {np.max(diff_scores):.4f}")
    print(f"  Mean diff score   : {np.mean(diff_scores):.4f}")

    # Save the selected frames
    os.makedirs(f"expression_test/{label}", exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(f"expression_test/{label}/frame_{i}.jpg", frame)

    # Save diff score plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3))
    plt.plot(diff_scores)
    plt.axvspan(start_t * 25, end_t * 25, alpha=0.3, color='red',
                label=f'Selected window ({start_t:.1f}s-{end_t:.1f}s)')
    plt.xlabel("Frame")
    plt.ylabel("Difference Score")
    plt.title(f"Frame Activity — {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"expression_test/{label}_activity.png")
    plt.close()
    print(f"  Saved frames and activity plot")


# Test on PA2 and PA3 samples from same subject
test_video("081617_m_27", "081617_m_27-PA2-001", "PA2")
test_video("081617_m_27", "081617_m_27-PA3-003", "PA3")

print("\nDone — open expression_test/ folder to see results")