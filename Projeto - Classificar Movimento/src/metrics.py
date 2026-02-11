from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from angle_utils import Point2D, angle_3points


LandmarkFrame = Dict[str, Optional[Point2D]]


@dataclass
class SeriesResult:
    elbow_angle_deg: List[Optional[float]]
    trunk_angle_deg: List[Optional[float]]
    wrist_to_shoulder_dist: List[Optional[float]]


# Suavização de séries temporais (média móvel simples), Mediapipe pode tremer um pouco. Isso ajuda a reduzir ruído e melhorar métricas. Mantém None quando não há dados suficientes.
def moving_average(values: List[Optional[float]], window: int = 5) -> List[Optional[float]]:

    if window <= 1:
        return values[:]

    half = window // 2
    out: List[Optional[float]] = []

    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        chunk = [v for v in values[start:end] if v is not None]

        if not chunk:
            out.append(None)
        else:
            out.append(sum(chunk) / len(chunk))

    return out

# Distância Euclidiana entre dois pontos 2D (linha reta)
def euclidean_dist(a: Point2D, b: Point2D) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def _min_max_amp(values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    mn = min(values)
    mx = max(values)
    amp = mx - mn
    return mn, mx, amp


def compute_series_from_landmarks(frames: List[LandmarkFrame], *, smooth_window: int = 5) -> SeriesResult:
    """
    Recebe uma lista de frames com landmarks e calcula séries por frame:
    - ângulo do cotovelo: angle(shoulder, elbow, wrist)
    - ângulo do tronco:   angle(shoulder, hip, knee)
    - distância punho->ombro: útil para detectar repetição na remada
    """
    elbow_angles: List[Optional[float]] = []
    trunk_angles: List[Optional[float]] = []
    wrist_shoulder: List[Optional[float]] = []

    for f in frames:
        shoulder = f.get("shoulder")
        elbow = f.get("elbow")
        wrist = f.get("wrist")
        hip = f.get("hip")
        knee = f.get("knee")

        # Cotovelo: A=ombro, B=cotovelo, C=punho (ângulo no cotovelo)
        if shoulder is None or elbow is None or wrist is None:
            elbow_angles.append(None)
            wrist_shoulder.append(None)
        else:
            elbow_angles.append(angle_3points(shoulder, elbow, wrist))
            wrist_shoulder.append(euclidean_dist(wrist, shoulder))

        # Tronco: A=ombro, B=quadril, C=joelho (ângulo no quadril)
        if shoulder is None or hip is None or knee is None:
            trunk_angles.append(None)
        else:
            trunk_angles.append(angle_3points(shoulder, hip, knee))

    # Suavização (reduz ruído/erro)
    elbow_angles_s = moving_average(elbow_angles, window=smooth_window)
    trunk_angles_s = moving_average(trunk_angles, window=smooth_window)
    wrist_shoulder_s = moving_average(wrist_shoulder, window=smooth_window)

    return SeriesResult(
        elbow_angle_deg=elbow_angles_s,
        trunk_angle_deg=trunk_angles_s,
        wrist_to_shoulder_dist=wrist_shoulder_s,
    )


def compute_global_metrics(series: SeriesResult) -> Dict[str, Optional[float]]:
    elbow_vals = [v for v in series.elbow_angle_deg if v is not None]
    trunk_vals = [v for v in series.trunk_angle_deg if v is not None]
    ws_vals = [v for v in series.wrist_to_shoulder_dist if v is not None]

    elbow_min, elbow_max, elbow_amp = _min_max_amp(elbow_vals)
    trunk_min, trunk_max, trunk_var = _min_max_amp(trunk_vals)
    ws_min, ws_max, ws_range = _min_max_amp(ws_vals)

    # média do tronco (postura)
    trunk_mean = (sum(trunk_vals) / len(trunk_vals)) if trunk_vals else None

    # desvio padrão do tronco (estabilidade mais robusta que só min/max)
    trunk_std = None
    if trunk_vals:
        m = trunk_mean
        trunk_std = (sum((x - m) ** 2 for x in trunk_vals) /
                     len(trunk_vals)) ** 0.5

    return {
        "elbow_min": elbow_min,
        "elbow_max": elbow_max,
        "elbow_amplitude": elbow_amp,

        "trunk_min": trunk_min,
        "trunk_max": trunk_max,
        "trunk_variation": trunk_var,
        "trunk_mean": trunk_mean,
        "trunk_std": trunk_std,

        "wrist_shoulder_min_dist": ws_min,
        "wrist_shoulder_max_dist": ws_max,
        "wrist_shoulder_range": ws_range,
    }
