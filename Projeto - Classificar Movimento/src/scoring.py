from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ScoreResult2:
    score_elbow: float           # 0..100
    score_trunk: float           # 0..100
    label_elbow: str             # ruim | medio | ok
    label_trunk: str             # ruim | medio | ok
    warnings_elbow: List[str]
    warnings_trunk: List[str]
    details: Dict[str, float]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score_from_range(value: float, good_min: float, good_max: float, falloff: float) -> float:
    """Dentro do intervalo -> 100, fora cai linearmente até 0."""
    if good_min <= value <= good_max:
        return 100.0
    dist = (good_min - value) if value < good_min else (value - good_max)
    return _clamp(100.0 * (1.0 - dist / falloff), 0.0, 100.0)


def _score_from_max(value: float, good_max: float, falloff: float) -> float:
    """Menor é melhor: <= good_max 100, acima cai até 0."""
    if value <= good_max:
        return 100.0
    dist = value - good_max
    return _clamp(100.0 * (1.0 - dist / falloff), 0.0, 100.0)


def _label(score: float, ok: float = 85.0, mid: float = 55.0) -> str:
    if score >= ok:
        return "ok"
    if score >= mid:
        return "medio"
    return "ruim"


def score_row_two_notes(
    global_metrics: Dict[str, Optional[float]],
    *,
    # ===== Cotovelo (amplitude)
    elbow_amp_good_min: float = 60.0,
    elbow_amp_good_max: float = 120.0,
    elbow_amp_falloff: float = 50.0,

    # ===== Tronco (postura via ângulo médio)
    trunk_mean_good_min: float = 80.0,
    trunk_mean_good_max: float = 100.0,
    trunk_mean_falloff: float = 30.0,

    # ===== Tronco (estabilidade via std)
    trunk_std_good_max: float = 18.0,
    trunk_std_falloff: float = 20.0,

    # ===== Como juntar postura+estabilidade na nota do tronco
    trunk_w_posture: float = 0.6,
    trunk_w_stability: float = 0.4,
) -> ScoreResult2:
    warnings_elbow: List[str] = []
    warnings_trunk: List[str] = []
    details: Dict[str, float] = {}

    elbow_amp = global_metrics.get("elbow_amplitude")
    trunk_mean = global_metrics.get("trunk_mean")
    trunk_std = global_metrics.get("trunk_std")

    # ----------------
    # NOTA DO COTOVELO
    # ----------------
    if elbow_amp is None:
        score_elbow = 0.0
        warnings_elbow.append("Não foi possível medir o cotovelo (pose falhou ou vídeo ruim).")
    else:
        score_elbow = _score_from_range(elbow_amp, elbow_amp_good_min, elbow_amp_good_max, elbow_amp_falloff)
        details["elbow_amplitude"] = float(elbow_amp)

        if elbow_amp < elbow_amp_good_min:
            warnings_elbow.append("Amplitude curta no cotovelo (puxou pouco).")
        elif elbow_amp > elbow_amp_good_max:
            warnings_elbow.append("Amplitude alta/inconsistente no cotovelo (execução ou detecção).")

    details["score_elbow"] = float(score_elbow)
    label_elbow = _label(score_elbow)

    # --------------
    # NOTA DO TRONCO
    # --------------
    # Postura
    if trunk_mean is None:
        score_trunk_posture = 100.0  # não penaliza se não mediu
        warnings_trunk.append("Não foi possível medir o ângulo médio do tronco (sem penalidade).")
    else:
        score_trunk_posture = _score_from_range(trunk_mean, trunk_mean_good_min, trunk_mean_good_max, trunk_mean_falloff)
        details["trunk_mean"] = float(trunk_mean)

        if trunk_mean < trunk_mean_good_min:
            warnings_trunk.append("Tronco muito inclinado (abaixo do ideal).")
        elif trunk_mean > trunk_mean_good_max:
            warnings_trunk.append("Tronco muito reto/aberto (acima do ideal).")

    # Estabilidade (boneco de posto)
    if trunk_std is None:
        score_trunk_stability = 100.0
        warnings_trunk.append("Não foi possível medir estabilidade do tronco (sem penalidade).")
    else:
        score_trunk_stability = _score_from_max(trunk_std, trunk_std_good_max, trunk_std_falloff)
        details["trunk_std"] = float(trunk_std)

        if trunk_std > (trunk_std_good_max + trunk_std_falloff * 0.60):
            warnings_trunk.append("Tronco indo e voltando demais (boneco de posto).")
        elif trunk_std > trunk_std_good_max:
            warnings_trunk.append("Tronco instável (controle melhor o balanço).")

    details["score_trunk_posture"] = float(score_trunk_posture)
    details["score_trunk_stability"] = float(score_trunk_stability)

    # Junta em uma nota única do tronco (mas separada do cotovelo!)
    score_trunk = (
        trunk_w_posture * score_trunk_posture
        + trunk_w_stability * score_trunk_stability
    )
    score_trunk = float(_clamp(score_trunk, 0.0, 100.0))

    details["score_trunk"] = float(score_trunk)
    label_trunk = _label(score_trunk)

    return ScoreResult2(
        score_elbow=float(score_elbow),
        score_trunk=float(score_trunk),
        label_elbow=label_elbow,
        label_trunk=label_trunk,
        warnings_elbow=warnings_elbow,
        warnings_trunk=warnings_trunk,
        details=details,
    )