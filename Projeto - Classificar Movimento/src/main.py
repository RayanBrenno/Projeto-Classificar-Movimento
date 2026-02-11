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
print("\n")
print("=" * 60)
print(f"VÍDEO ANALISADO: {video_path}")
print("=" * 60)

print(f"COTOVELO: {result.score_elbow:.1f} ({result.label_elbow})")
if result.warnings_elbow:
    for w in result.warnings_elbow:
        print(f" - {w}")
else:
    print(" - Sem feedback negativo")

print()

print(f"TRONCO:   {result.score_trunk:.1f} ({result.label_trunk})")
if result.warnings_trunk:
    for w in result.warnings_trunk:
        print(f" - {w}")
else:
    print(" - Sem feedback negativo")

print("=" * 60)
