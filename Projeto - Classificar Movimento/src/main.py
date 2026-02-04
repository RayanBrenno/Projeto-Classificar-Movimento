from pose_extraction import extract_landmarks_from_video
from metrics import compute_series_from_landmarks, compute_global_metrics
from scoring import score_row_two_notes

video_path = "data/raw/certo.mp4"

frames = extract_landmarks_from_video(
    video_path=video_path,
    side="right",
    visibility_threshold=0.5,
    save_csv=True,
    csv_path="outputs/landmarks_certo.csv",
    save_annotated_video=True,
    annotated_video_path="outputs/certo.mp4"
)

series = compute_series_from_landmarks(frames, smooth_window=5)
gm = compute_global_metrics(series)

result = score_row_two_notes(gm)

print("Métricas:", gm)
print("Cotovelo:", result.score_elbow, result.label_elbow, result.warnings_elbow)
print("Tronco:", result.score_trunk, result.label_trunk, result.warnings_trunk)
print("Details:", result.details)