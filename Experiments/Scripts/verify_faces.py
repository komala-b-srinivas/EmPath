import os

FACES_DIR = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/faces_v3"

for split in ["train", "val", "test"]:
    for cls in ["PA2", "PA3"]:
        folder = os.path.join(FACES_DIR, split, cls)
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
            print(f"{split}/{cls}: {count} images")
        else:
            print(f"{split}/{cls}: FOLDER NOT FOUND")

