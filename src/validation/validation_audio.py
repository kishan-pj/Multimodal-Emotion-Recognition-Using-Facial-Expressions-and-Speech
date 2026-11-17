
import av
import pandas as pd
from pathlib import Path

# ==========================================
# 1. Define project directories
# ==========================================

project_path = Path(__file__).resolve().parents[2]

data_path = project_path / "data" / "raw" / "MELD"

report_path = project_path / "reports" / "validation"
report_path.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Define dataset files and folders
# ==========================================

datasets = {
    "Train": ("train_sent_emo.csv", "train"),
    "Validation": ("dev_sent_emo.csv", "dev"),
    "Test": ("test_sent_emo.csv", "test")
}


# ==========================================
# 3. Function to validate audio
# ==========================================

def validate_audio(video_path):

    try:
        with av.open(str(video_path)) as container:

            # Check whether audio stream exists
            if not container.streams.audio:
                return "No audio stream"

            audio_stream = container.streams.audio[0]

            # Check sample rate
            sample_rate = audio_stream.codec_context.sample_rate

            if not sample_rate or sample_rate <= 0:
                return "Invalid sample rate"

            # Try to decode the first audio frame
            audio_frame = next(
                container.decode(audio=0),
                None
            )

            if audio_frame is None or audio_frame.samples <= 0:
                return "Cannot decode audio frame"

    except Exception as error:
        return f"Cannot open or decode audio: {error}"

    return None


# ==========================================
# 4. Validate MELD audio
# ==========================================

validation_results = []
audio_issues = []

for dataset_name, (csv_file, folder_name) in datasets.items():

    print(f"\n===== {dataset_name.upper()} AUDIO VALIDATION =====")

    df = pd.read_csv(data_path / csv_file)

    video_folder = data_path / folder_name

    if not video_folder.is_dir():
        raise FileNotFoundError(video_folder)

    # Match actual videos by filename
    # Ignore macOS metadata files
    video_lookup = {}

    for video_path in video_folder.rglob("*.mp4"):

        if video_path.name.startswith("._"):
            continue

        video_name = video_path.stem.lower()

        video_lookup.setdefault(video_name, []).append(video_path)

    valid_count = 0
    invalid_count = 0

    for index, row in df.iterrows():

        dialogue_id = int(row["Dialogue_ID"])
        utterance_id = int(row["Utterance_ID"])

        video_name = f"dia{dialogue_id}_utt{utterance_id}"
        matching_files = video_lookup.get(video_name.lower(), [])

        # Check availability
        if not matching_files:
            issue = "Missing video"

        elif len(matching_files) > 1:
            issue = "Duplicate video filename"

        else:
            issue = validate_audio(matching_files[0])

        if issue:
            invalid_count += 1

            audio_issues.append({
                "Dataset": dataset_name,
                "Video": video_name + ".mp4",
                "Issue": issue
            })

        else:
            valid_count += 1

        if (index + 1) % 1000 == 0:
            print(f"Checked {index + 1} utterances")

    print("Total annotations:", len(df))
    print("Passed audio checks:", valid_count)
    print("Flagged records:", invalid_count)

    validation_results.append({
        "Dataset": dataset_name,
        "Total_Annotations": len(df),
        "Passed_Audio_Checks": valid_count,
        "Flagged_Records": invalid_count
    })


# ==========================================
# 5. Save validation reports
# ==========================================

summary_df = pd.DataFrame(validation_results)

issues_df = pd.DataFrame(
    audio_issues,
    columns=["Dataset", "Video", "Issue"]
)

summary_df.to_csv(
    report_path / "audio_validation_summary.csv",
    index=False
)

issues_df.to_csv(
    report_path / "audio_issues.csv",
    index=False
)


# ==========================================
# 6. Display final results
# ==========================================

print("\n===== AUDIO VALIDATION SUMMARY =====")
print(summary_df.to_string(index=False))

print("\n===== AUDIO ISSUE COUNTS =====")

if issues_df.empty:
    print("No audio issues found.")
else:
    print(issues_df.groupby(["Dataset", "Issue"]).size())

print("\nAudio validation completed!")
print("Reports saved to:", report_path)
