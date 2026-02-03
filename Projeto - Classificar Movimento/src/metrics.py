
"""
Métricas para avaliação do exercício a partir de landmarks 2D (pose).

Este módulo NÃO lê vídeo e NÃO roda MediaPipe.
Ele recebe séries temporais (por frame) de pontos (x,y) e devolve:
- ângulos por frame (cotovelo, tronco)
- amplitudes e variações
- segmentação simples em repetições (opcional)
- métricas por repetição

Dependências internas:
- src/angle_utils.py  (Point2D, angle_3points, angular_variation)

Observação:
- Para vídeo lateral, cotovelo e tronco são as métricas mais confiáveis.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from angle_utils import Point2D, angle_3points, angular_variation


# -----------------------------
# Tipos de dados
# -----------------------------

# Um frame pode ter alguns landmarks (ombro, cotovelo, punho, quadril, joelho, etc.)
# Em cada frame, alguns podem faltar (pose falhou) -> valor None.
# ex: {"shoulder": Point2D(...), "elbow": ..., ...}
LandmarkFrame = Dict[str, Optional[Point2D]]


@dataclass
class SeriesResult:
    """Séries temporais calculadas por frame."""
    elbow_angle_deg: List[Optional[float]]
    trunk_angle_deg: List[Optional[float]]
    # útil para detectar repetições
    wrist_to_shoulder_dist: List[Optional[float]]


@dataclass
class RepSegment:
    """Representa um segmento de frames [start, end] (inclusive) que forma 1 repetição."""
    start: int
    end: int


@dataclass
class RepMetrics:
    """Métricas calculadas dentro de uma repetição."""
    rep_index: int
    frames: RepSegment

    elbow_min: Optional[float]
    elbow_max: Optional[float]
    elbow_amplitude: Optional[float]

    trunk_min: Optional[float]
    trunk_max: Optional[float]
    trunk_variation: Optional[float]

    # Métrica auxiliar para remada: quanto o punho se aproximou do ombro
    wrist_shoulder_min_dist: Optional[float]
    wrist_shoulder_max_dist: Optional[float]
    wrist_shoulder_range: Optional[float]


def moving_average(values: List[Optional[float]], window: int = 5) -> List[Optional[float]]:
    """
    Suaviza uma série com média móvel simples.

    - Mantém None quando não há dados suficientes.
    - Para cada posição i, calcula média dos valores válidos dentro da janela.

    Por que isso ajuda?
    - MediaPipe/pose pode tremer um pouco. A suavização reduz ruído e melhora métricas.
    """
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


def euclidean_dist(a: Point2D, b: Point2D) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def _slice_valid(values: List[Optional[float]], start: int, end: int) -> List[float]:
    """Extrai valores válidos (não None) no intervalo [start, end]."""
    return [v for v in values[start:end + 1] if v is not None]


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

    Se algum ponto necessário faltar num frame, o resultado desse frame vira None.
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

    # Suavização (reduz ruído)
    elbow_angles_s = moving_average(elbow_angles, window=smooth_window)
    trunk_angles_s = moving_average(trunk_angles, window=smooth_window)
    wrist_shoulder_s = moving_average(wrist_shoulder, window=smooth_window)

    return SeriesResult(
        elbow_angle_deg=elbow_angles_s,
        trunk_angle_deg=trunk_angles_s,
        wrist_to_shoulder_dist=wrist_shoulder_s,
    )


def compute_global_metrics(series: SeriesResult) -> Dict[str, Optional[float]]:
    """
    Calcula métricas para o vídeo todo (sem separar em repetições).
    Bom para o MVP inicial.

    Retorna:
    - elbow_min, elbow_max, elbow_amplitude
    - trunk_min, trunk_max, trunk_variation
    - wrist_shoulder_min_dist, wrist_shoulder_max_dist, wrist_shoulder_range
    """
    elbow_vals = [v for v in series.elbow_angle_deg if v is not None]
    trunk_vals = [v for v in series.trunk_angle_deg if v is not None]
    ws_vals = [v for v in series.wrist_to_shoulder_dist if v is not None]

    elbow_min, elbow_max, elbow_amp = _min_max_amp(elbow_vals)
    trunk_min, trunk_max, trunk_var = _min_max_amp(trunk_vals)
    ws_min, ws_max, ws_range = _min_max_amp(ws_vals)

    return {
        "elbow_min": elbow_min,
        "elbow_max": elbow_max,
        "elbow_amplitude": elbow_amp,

        "trunk_min": trunk_min,
        "trunk_max": trunk_max,
        "trunk_variation": trunk_var,

        "wrist_shoulder_min_dist": ws_min,
        "wrist_shoulder_max_dist": ws_max,
        "wrist_shoulder_range": ws_range,
    }


def detect_reps_by_wrist_shoulder_distance(wrist_shoulder_dist: List[Optional[float]], *, min_frames_per_rep: int = 15, prominence: float = 0.02) -> List[RepSegment]:
    """
    Detecta repetições usando a curva de distância punho->ombro.

    Intuição (remada):
    - Quando puxa, punho aproxima do ombro -> distância diminui.
    - Quando retorna, distância aumenta.
    Assim, cada repetição costuma formar um "vale" (mínimo) e "picos" (máximos).

    Estratégia simples (sem scipy):
    - achar mínimos locais "fortes" (vales) com uma noção simples de proeminência
    - criar segmentos de repetições entre picos ou entre mínimos consecutivos

    Parâmetros:
    - min_frames_per_rep: evita detectar reps minúsculas (ruído)
    - prominence: quanto o vale precisa "descer" em relação às vizinhanças (depende do seu vídeo e normalização)

    Observação:
    - Isso é um detector simples para o MVP+. Você pode melhorar depois.
    """
    # Converter None -> pular
    vals = wrist_shoulder_dist

    # Encontrar mínimos locais candidatos
    minima: List[int] = []
    for i in range(1, len(vals) - 1):
        if vals[i] is None or vals[i - 1] is None or vals[i + 1] is None:
            continue
        if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            # proeminência simples: diferença entre vizinhos médios e o ponto
            neighbor_avg = (vals[i - 1] + vals[i + 1]) / 2
            if (neighbor_avg - vals[i]) >= prominence:
                minima.append(i)

    if len(minima) < 2:
        return []

    # Transformar mínimos em segmentos: [min_k, min_{k+1}]
    reps: List[RepSegment] = []
    for k in range(len(minima) - 1):
        start = minima[k]
        end = minima[k + 1]
        if (end - start) >= min_frames_per_rep:
            reps.append(RepSegment(start=start, end=end))

    return reps


def compute_rep_metrics(series: SeriesResult, reps: List[RepSegment]) -> List[RepMetrics]:
    """
    Calcula métricas para cada repetição detectada.
    """
    results: List[RepMetrics] = []

    for idx, seg in enumerate(reps):
        elbow_seg = _slice_valid(series.elbow_angle_deg, seg.start, seg.end)
        trunk_seg = _slice_valid(series.trunk_angle_deg, seg.start, seg.end)
        ws_seg = _slice_valid(
            series.wrist_to_shoulder_dist, seg.start, seg.end)

        e_min, e_max, e_amp = _min_max_amp(elbow_seg)
        t_min, t_max, t_var = _min_max_amp(trunk_seg)
        ws_min, ws_max, ws_rng = _min_max_amp(ws_seg)

        results.append(
            RepMetrics(
                rep_index=idx,
                frames=seg,

                elbow_min=e_min,
                elbow_max=e_max,
                elbow_amplitude=e_amp,

                trunk_min=t_min,
                trunk_max=t_max,
                trunk_variation=t_var,

                wrist_shoulder_min_dist=ws_min,
                wrist_shoulder_max_dist=ws_max,
                wrist_shoulder_range=ws_rng,
            )
        )

    return results
