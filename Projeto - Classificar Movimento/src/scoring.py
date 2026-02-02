# src/scoring.py
"""
Converte métricas (vindas do metrics.py) em:
- score final (0..100)
- classificação textual (errado / meio certo / ok)
- mensagens de feedback

MVP: usa principalmente amplitude do cotovelo.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ScoreResult:
    score: float                 # 0..100
    label: str                   # "errado" | "meio certo" | "ok"
    warnings: List[str]          # mensagens
    details: Dict[str, float]    # notas parciais / valores usados


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score_from_range(value: float, good_min: float, good_max: float) -> float:
    """
    Converte um valor numérico em uma nota 0..100.
    - Dentro de [good_min, good_max] -> 100
    - Fora -> cai linearmente até 0 com base na distância até o intervalo

    Exemplo:
      amplitude ideal do cotovelo: 60..110 graus
    """
    if good_min <= value <= good_max:
        return 100.0

    # distância até o intervalo
    if value < good_min:
        dist = good_min - value
    else:
        dist = value - good_max

    # escala de queda (ajustável): quanto "fora" já vira 0
    # Se dist >= falloff, nota vira 0
    falloff = 50.0
    score = 100.0 * (1.0 - (dist / falloff))
    return _clamp(score, 0.0, 100.0)


def score_row_mvp(
    global_metrics: Dict[str, Optional[float]],
    *,
    elbow_amp_good_min: float = 60.0,
    elbow_amp_good_max: float = 120.0,
) -> ScoreResult:
    """
    MVP da remada:
    - usa elbow_amplitude como critério principal.
    Retorna score + label + mensagens.

    Parâmetros elbow_amp_good_min/max são limites iniciais (você ajusta depois com vídeos reais).
    """
    warnings: List[str] = []
    details: Dict[str, float] = {}

    elbow_amp = global_metrics.get("elbow_amplitude")

    # Se não deu pra medir, retorna "não avaliável"
    if elbow_amp is None:
        return ScoreResult(
            score=0.0,
            label="errado",
            warnings=["Não foi possível medir o cotovelo (pose falhou ou vídeo ruim)."],
            details={},
        )

    # Nota do cotovelo
    elbow_score = _score_from_range(elbow_amp, elbow_amp_good_min, elbow_amp_good_max)
    details["elbow_amplitude"] = float(elbow_amp)
    details["elbow_score"] = float(elbow_score)

    # Regras de mensagem
    if elbow_amp < elbow_amp_good_min:
        warnings.append("Amplitude de movimento curta (cotovelo flexionou pouco).")
    elif elbow_amp > elbow_amp_good_max:
        warnings.append("Amplitude muito alta/inconsistente (verifique detecção ou execução).")

    # Score final (MVP: só cotovelo)
    final_score = elbow_score

    # Label (faixas simples)
    if final_score >= 85:
        label = "ok"
    elif final_score >= 55:
        label = "meio certo"
    else:
        label = "errado"

    return ScoreResult(
        score=float(final_score),
        label=label,
        warnings=warnings,
        details=details,
    )
