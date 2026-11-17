
import cv2
import pandas as pd
import math
from pathlib import Path

# ==========================================
# 1. Define project directories
# ==========================================

project_path = Path(__file__).resolve().parents[2]

data_path = project_path / "data" / "raw" / "MELD"

report_path = project_path / "reports" / "validation"
report_path.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Define dataset folders
# ==========================================

dataset_folders = {
    "Train": "train",
    "Validation": "dev",
    "Test": "test"
}


# ==========================================
# 3. Function to validate each video
# ==========================================

def validate_video(video_path):

    issues = []

    cap = cv2.VideoCapture(str(video_path))

    try:
        # Check whether video opens
        if not cap.isOpened():
            return ["Cannot open video"]

        # Read video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Validate FPS
        if not math.isfinite(fps) or fps <= 0:
            issues.append("Invalid FPS")

        # Validate frame count
        if not math.isfinite(frame_count) or frame_count <= 0:
            issues.append("Invalid frame count")

        # Validate resolution
        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 0
            or height <= 0
        ):
            issues.append("Invalid resolution")

        # Check whether first frame can be read
        success, frame = cap.read()

        if not success or frame is None:
            issues.append("Cannot read first frame")

    finally:
        cap.release()

    return issues


# ==========================================
# 4. Validate all MELD videos
# ==========================================

validation_results = []
invalid_videos = []

for dataset_name, folder_name in dataset_folders.items():

    print(f"\n===== {dataset_name.upper()} VIDEO VALIDATION =====")

    video_folder = data_path / folder_name

    if not video_folder.is_dir():
        raise FileNotFoundError(
            f"Video folder not found: {video_folder}"
        )

    # Find videos inside nested folders
    
    # Find actual MP4 videos and ignore macOS metadata files
    video_files = sorted(
        file for file in video_folder.rglob("*.mp4")
        if not file.name.startswith("._")
    )


    valid_count = 0
    invalid_count = 0

    print("Total video files:", len(video_files))

    for index, video_path in enumerate(video_files, start=1):

        issues = validate_video(video_path)

        if issues:
            invalid_count += 1

            invalid_videos.append({
                "Dataset": dataset_name,
                "Video": video_path.name,
                "Issue": "; ".join(issues)
            })

        else:
            valid_count += 1

        # Display progress
        if index % 1000 == 0:
            print(f"Checked {index} videos")

    print("Valid videos:", valid_count)
    print("Invalid videos:", invalid_count)

    validation_results.append({
        "Dataset": dataset_name,
        "Total_Videos": len(video_files),
        "Valid_Videos": valid_count,
        "Invalid_Videos": invalid_count
    })


# ==========================================
# 5. Save validation reports
# ==========================================

summary_df = pd.DataFrame(validation_results)

invalid_df = pd.DataFrame(
    invalid_videos,
    columns=["Dataset", "Video", "Issue"]
)

summary_df.to_csv(
    report_path / "video_integrity_summary.csv",
    index=False
)

invalid_df.to_csv(
    report_path / "invalid_videos.csv",
    index=False
)


# ==========================================
# 6. Display final summary
# ==========================================

print("\n===== VIDEO VALIDATION SUMMARY =====")

print(summary_df.to_string(index=False))

print("\nVideo integrity validation completed!")

print("Reports saved to:", report_path)
