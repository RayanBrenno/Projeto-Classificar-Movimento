from angle_utils import Point2D
from metrics import compute_series_from_landmarks, compute_global_metrics
from scoring import score_row_mvp


def build_fake_frames():
    """
    Simula 8 frames de uma remada:
    - 3 indo
    - 2 no meio
    - 3 voltando
    """

    hip = Point2D(0.50, 0.55)
    knee = Point2D(0.50, 0.80)
    elbow = Point2D(0.55, 0.45)

    shoulders = [
        Point2D(0.50, 0.30),
        Point2D(0.50, 0.30),
        Point2D(0.495, 0.30),
        Point2D(0.49, 0.30),
        Point2D(0.49, 0.30),
        Point2D(0.495, 0.30),
        Point2D(0.50, 0.30),
        Point2D(0.50, 0.30),
    ]

    wrists = [
        Point2D(0.7273, 0.4187),
        Point2D(0.6879, 0.3343),
        Point2D(0.6116, 0.2809),
        Point2D(0.5500, 0.2700),
        Point2D(0.5657, 0.2707),
        Point2D(0.6116, 0.2809),
        Point2D(0.6879, 0.3343),
        Point2D(0.7273, 0.4187),
    ]

    frames = []
    for i in range(8):
        frames.append({
            "shoulder": shoulders[i],
            "elbow": elbow,
            "wrist": wrists[i],
            "hip": hip,
            "knee": knee,
        })

    return frames


def main():
    print("▶️ Criando frames simulados...")
    frames = build_fake_frames()

    print("▶️ Calculando métricas...")
    series = compute_series_from_landmarks(frames, smooth_window=1)
    global_metrics = compute_global_metrics(series)

    print("▶️ Calculando score...")
    score = score_row_mvp(global_metrics)

    print("\n=========== RESULTADO ===========")
    print("Séries:")
    print("Ângulo cotovelo:", series.elbow_angle_deg)
    print("Ângulo tronco:  ", series.trunk_angle_deg)
    print("Dist punho-ombro:", series.wrist_to_shoulder_dist)

    print("\nMétricas globais:")
    for k, v in global_metrics.items():
        print(f"{k}: {v}")

    print("\nScore final:")
    print(f"Score: {score.score:.1f}%")
    print(f"Label: {score.label}")
    if score.warnings:
        print("Avisos:")
        for w in score.warnings:
            print(" -", w)

    print("================================\n")


if __name__ == "__main__":
    main()
