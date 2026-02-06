from pose_extraction import extract_landmarks_from_video
from metrics import compute_series_from_landmarks, compute_global_metrics
from scoring import score_row_two_notes

video_path = "data/raw/errado2.mp4"

frames = extract_landmarks_from_video(
    video_path=video_path,
    side="right",
    visibility_threshold=0.5,
    save_csv=True,
    csv_path="outputs/landmarks_errado2.csv",
    save_annotated_video=True,
    annotated_video_path="outputs/errado2.mp4"
)

series = compute_series_from_landmarks(frames, smooth_window=5)
gm = compute_global_metrics(series)

result = score_row_two_notes(gm)

print("Cotovelo:", result.score_elbow, result.label_elbow, "\n" ,result.warnings_elbow)
print("Tronco:", result.score_trunk, result.label_trunk , "\n", result.warnings_trunk)