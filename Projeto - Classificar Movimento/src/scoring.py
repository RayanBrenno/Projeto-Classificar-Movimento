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
    # =========================
    # COTOVELO: amplitude (ROM)
    # =========================
    elbow_amp_good_min: float = 100.0,
    elbow_amp_good_max: float = 120.0,
    elbow_amp_falloff: float = 25.0,

    # =========================
    # COTOVELO: "assinatura" do movimento (elbow_min)
    # Perfeito que você passou: 59.67
    # =========================
    elbow_min_target: float = 59.67,
    elbow_min_tol: float = 4.0,         # folga pequena (punitivo)
    elbow_min_falloff: float = 8.0,     # queda rápida (punitivo)

    # Como juntar amplitude + elbow_min na nota do cotovelo
    elbow_w_amp: float = 0.55,
    elbow_w_min: float = 0.45,

    # =========================
    # TRONCO: variação (max-min)
    # Seu "perfeito" ≈ 41.86, então bom_max precisa cobrir isso
    # =========================
    trunk_var_good_max: float = 45.0,
    trunk_var_falloff: float = 25.0,

    # =========================
    # TRONCO: postura via ângulo médio (fallback)
    # =========================
    trunk_mean_good_min: float = 85.0,
    trunk_mean_good_max: float = 105.0,
    trunk_mean_falloff: float = 25.0,

    # =========================
    # TRONCO: estabilidade via std
    # Seu "perfeito" ≈ 14.62
    # =========================
    trunk_std_good_max: float = 18.0,
    trunk_std_falloff: float = 20.0,

    # =========================
    # TRONCO: punir quando trunk_max abre demais
    # Perfeito que você passou: 112.73
    # =========================
    trunk_max_target: float = 112.73,
    trunk_max_tol: float = 4.0,         # folga pequena (punitivo)
    trunk_max_falloff: float = 10.0,    # queda rápida se abrir demais
    trunk_w_max: float = 0.25,          # peso do trunk_max na nota do tronco

    # Como juntar (postura/variação + estabilidade) antes de aplicar trunk_max
    trunk_w_posture: float = 0.65,
    trunk_w_stability: float = 0.35,

    # =========================
    # Extra: mão vindo (proxy)
    # =========================
    wrist_shoulder_range_good_min: float = 0.16,
    wrist_shoulder_range_good_max: float = 0.22,
    wrist_shoulder_range_falloff: float = 0.08,
    wrist_hint_weight: float = 0.0,  # deixe 0.0 se não quiser entrar na nota (só feedback)
) -> ScoreResult2:
    warnings_elbow: List[str] = []
    warnings_trunk: List[str] = []
    details: Dict[str, float] = {}

    elbow_min = global_metrics.get("elbow_min")
    elbow_amp = global_metrics.get("elbow_amplitude")

    trunk_var = global_metrics.get("trunk_variation")
    trunk_mean = global_metrics.get("trunk_mean")
    trunk_std = global_metrics.get("trunk_std")
    trunk_max = global_metrics.get("trunk_max")

    wrist_shoulder_range = global_metrics.get("wrist_shoulder_range")

    # ----------------
    # NOTA DO COTOVELO
    # ----------------
    # 1) amplitude
    if elbow_amp is None:
        score_elbow_amp = 0.0
        warnings_elbow.append("Não foi possível medir a amplitude do cotovelo (pose falhou ou vídeo ruim).")
    else:
        score_elbow_amp = _score_from_range(elbow_amp, elbow_amp_good_min, elbow_amp_good_max, elbow_amp_falloff)
        details["elbow_amplitude"] = float(elbow_amp)

        if elbow_amp < elbow_amp_good_min:
            warnings_elbow.append("Amplitude curta no cotovelo (puxou pouco / encurtou a puxada).")
        elif elbow_amp > elbow_amp_good_max:
            warnings_elbow.append("Amplitude alta/inconsistente no cotovelo (pode ser detecção ou execução irregular).")

    # 2) elbow_min (punitivo, folga pequena)
    if elbow_min is None:
        score_elbow_min = 70.0  # não zera, mas faltou dado
        warnings_elbow.append("Não foi possível medir o ângulo mínimo do cotovelo (sem penalidade total).")
    else:
        good_min = elbow_min_target - elbow_min_tol
        good_max = elbow_min_target + elbow_min_tol
        score_elbow_min = _score_from_range(elbow_min, good_min, good_max, elbow_min_falloff)
        details["elbow_min"] = float(elbow_min)

        # Feedbacks direcionados (do jeito que você pediu)
        if elbow_min < good_min:
            warnings_elbow.append(
                "A puxada está subindo pro peito. Puxe no sentido da cintura (cotovelos indo pra trás e pra baixo)."
            )
        elif elbow_min > good_max:
            warnings_elbow.append(
                "Pouca contração no final da puxada. Contraia mais e finalize trazendo o cotovelo mais pra trás."
            )

    # 3) dica com wrist_shoulder_range (proxy do quanto a mão “viajou”)
    score_wrist_hint = 100.0
    if wrist_shoulder_range is not None:
        details["wrist_shoulder_range"] = float(wrist_shoulder_range)
        score_wrist_hint = _score_from_range(
            wrist_shoulder_range,
            wrist_shoulder_range_good_min,
            wrist_shoulder_range_good_max,
            wrist_shoulder_range_falloff,
        )

        if wrist_shoulder_range < wrist_shoulder_range_good_min:
            warnings_elbow.append("A mão quase não se aproxima do tronco. Faça uma puxada mais completa e controlada.")
        elif wrist_shoulder_range > wrist_shoulder_range_good_max:
            warnings_elbow.append("A mão está vindo demais (pode estar passando do ponto e gerando compensações).")

    # Junta nota do cotovelo
    score_elbow = (elbow_w_amp * score_elbow_amp) + (elbow_w_min * score_elbow_min)
    if wrist_hint_weight > 0.0:
        score_elbow = (1.0 - wrist_hint_weight) * score_elbow + wrist_hint_weight * score_wrist_hint

    score_elbow = float(_clamp(score_elbow, 0.0, 100.0))
    details["score_elbow_amp"] = float(score_elbow_amp)
    details["score_elbow_min"] = float(score_elbow_min)
    details["score_elbow"] = float(score_elbow)
    label_elbow = _label(score_elbow)

    # --------------
    # NOTA DO TRONCO
    # --------------
    # 1) Postura/roubo: prioriza trunk_variation. Se não existir, usa trunk_mean (fallback).
    if trunk_var is not None:
        score_trunk_posture = _score_from_max(trunk_var, trunk_var_good_max, trunk_var_falloff)
        details["trunk_variation"] = float(trunk_var)

        if trunk_var > (trunk_var_good_max + trunk_var_falloff * 0.60):
            warnings_trunk.append("Roubo forte com o tronco (muito balanço). Trave o core e mantenha o tronco mais estável.")
        elif trunk_var > trunk_var_good_max:
            warnings_trunk.append("Tronco mexendo além do ideal. Reduza o balanço e faça a força com costas e braços.")
    else:
        if trunk_mean is None:
            score_trunk_posture = 100.0
            warnings_trunk.append("Não foi possível medir o tronco (postura) (sem penalidade).")
        else:
            score_trunk_posture = _score_from_range(trunk_mean, trunk_mean_good_min, trunk_mean_good_max, trunk_mean_falloff)
            details["trunk_mean"] = float(trunk_mean)

            if trunk_mean < trunk_mean_good_min:
                warnings_trunk.append("Tronco muito inclinado (abaixo do ideal).")
            elif trunk_mean > trunk_mean_good_max:
                warnings_trunk.append("Tronco muito reto/aberto (acima do ideal).")

    # 2) Estabilidade: std
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

    # 3) Controle de "abrir demais": trunk_max (pune só quando passar do ideal)
    if trunk_max is None:
        score_trunk_max = 100.0
        warnings_trunk.append("Não foi possível medir o ângulo máximo do tronco (sem penalidade).")
    else:
        details["trunk_max"] = float(trunk_max)

        good_max = trunk_max_target + trunk_max_tol
        score_trunk_max = _score_from_max(trunk_max, good_max, trunk_max_falloff)

        if trunk_max > (good_max + trunk_max_falloff * 0.60):
            warnings_trunk.append("Tronco abrindo demais no final (jogando o corpo). Trave o core e finalize sem inclinar.")
        elif trunk_max > good_max:
            warnings_trunk.append("Tronco abrindo mais que o ideal. Controle o movimento e evite compensar com o corpo.")

    details["score_trunk_posture"] = float(score_trunk_posture)
    details["score_trunk_stability"] = float(score_trunk_stability)
    details["score_trunk_max"] = float(score_trunk_max)

    # Junta nota do tronco:
    # - primeiro combina postura + estabilidade (pesos internos trunk_w_posture/trunk_w_stability)
    # - depois mistura com trunk_max (peso trunk_w_max)
    w_max = float(_clamp(trunk_w_max, 0.0, 0.7))
    w_rest = 1.0 - w_max

    score_trunk_base = (trunk_w_posture * score_trunk_posture) + (trunk_w_stability * score_trunk_stability)
    score_trunk = (w_rest * score_trunk_base) + (w_max * score_trunk_max)
    score_trunk = float(_clamp(score_trunk, 0.0, 100.0))

    details["score_trunk_base"] = float(score_trunk_base)
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