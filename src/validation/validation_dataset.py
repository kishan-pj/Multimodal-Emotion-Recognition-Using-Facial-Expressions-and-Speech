
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
# 2. Define valid labels and columns
# ==========================================

valid_emotions = {
    "neutral", "joy", "sadness", "anger",
    "fear", "surprise", "disgust"
}

valid_sentiments = {
    "positive", "negative", "neutral"
}

required_columns = [
    "Dialogue_ID",
    "Utterance_ID",
    "Utterance",
    "Speaker",
    "Emotion",
    "Sentiment"
]


# ==========================================
# 3. Define dataset files
# ==========================================

dataset_files = {
    "Train": "train_sent_emo.csv",
    "Validation": "dev_sent_emo.csv",
    "Test": "test_sent_emo.csv"
}

validation_results = []


# ==========================================
# 4. Validate each dataset
# ==========================================

for dataset_name, filename in dataset_files.items():

    print(f"\n===== {dataset_name.upper()} DATASET =====")

    file_path = data_path / filename

    df = pd.read_csv(file_path)

    print("Total records:", len(df))

    # Check required columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    print("Missing required columns:", missing_columns)

    if missing_columns:
        print("Cannot continue validation for this dataset.")
        continue

    # Check missing values
    missing_values = df.isnull().sum().sum()
    print("Total missing values:", missing_values)

    # Check duplicate utterance identifiers
    duplicate_records = df.duplicated(
        subset=["Dialogue_ID", "Utterance_ID"]
    ).sum()

    print("Duplicate utterance IDs:", duplicate_records)

    # Check invalid emotion labels
    invalid_emotions = (
        df["Emotion"].notna()
        & ~df["Emotion"].isin(valid_emotions)
    ).sum()

    print("Invalid emotion labels:", invalid_emotions)

    # Check invalid sentiment labels
    invalid_sentiments = (
        df["Sentiment"].notna()
        & ~df["Sentiment"].isin(valid_sentiments)
    ).sum()

    print("Invalid sentiment labels:", invalid_sentiments)

    # Check empty or whitespace-only utterances
    empty_utterances = (
        df["Utterance"].fillna("").astype(str).str.strip() == ""
    ).sum()

    print("Empty utterances:", empty_utterances)

    # Store validation summary
    validation_results.append({
        "Dataset": dataset_name,
        "Total_Records": len(df),
        "Missing_Values": int(missing_values),
        "Duplicate_Ids": int(duplicate_records),
        "Invalid_Emotions": int(invalid_emotions),
        "Invalid_Sentiments": int(invalid_sentiments),
        "Empty_Utterances": int(empty_utterances)
    })


# ==========================================
# 5. Save validation report
# ==========================================

summary_df = pd.DataFrame(validation_results)

print("\n===== VALIDATION SUMMARY =====")
print(summary_df.to_string(index=False))

summary_df.to_csv(
    report_path / "annotation_validation_summary.csv",
    index=False
)

print("\nAnnotation validation completed!")
print("Report saved to:", report_path)



# ==========================================
# 6. Validate video file availability
# ==========================================

print("\n===== VIDEO FILE VALIDATION =====")

split_folders = {
    "Train": "train",
    "Validation": "dev",
    "Test": "test"
}

missing_videos = []

for dataset_name, filename in dataset_files.items():

    print(f"\n{dataset_name.upper()} VIDEOS")

    # Load the annotations
    df = pd.read_csv(data_path / filename)

    # Locate videos, including nested folders
    video_folder = data_path / split_folders[dataset_name]
    video_files = list(video_folder.rglob("*.mp4"))

    # Create lookup using video filenames
    video_names = {video.stem.lower() for video in video_files}

    missing_count = 0

    for _, row in df.iterrows():

        dialogue_id = int(row["Dialogue_ID"])
        utterance_id = int(row["Utterance_ID"])

        expected_name = f"dia{dialogue_id}_utt{utterance_id}"

        if expected_name.lower() not in video_names:
            missing_count += 1

            missing_videos.append({
                "Dataset": dataset_name,
                "Dialogue_ID": dialogue_id,
                "Utterance_ID": utterance_id,
                "Expected_Video": expected_name + ".mp4"
            })

    print("Annotated utterances:", len(df))
    print("Video files found:", len(video_files))
    print("Missing videos:", missing_count)


# Save missing video report
missing_df = pd.DataFrame(
    missing_videos,
    columns=[
        "Dataset",
        "Dialogue_ID",
        "Utterance_ID",
        "Expected_Video"
    ]
)

missing_df.to_csv(
    report_path / "missing_videos.csv",
    index=False
)

print("\nVideo availability validation completed!")
