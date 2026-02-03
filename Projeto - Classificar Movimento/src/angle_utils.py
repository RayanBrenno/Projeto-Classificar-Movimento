# src/angle_utils.py
from __future__ import annotations
from dataclasses import dataclass
from math import acos, degrees, sqrt
from typing import Iterable, List, Tuple, Optional


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

# Produto Escalar
def _dot(u: Point2D, v: Point2D) -> float:
    return u.x * v.x + u.y * v.y

# Tamanho (norma) do vetor
def _norm(u: Point2D) -> float:
    return sqrt(u.x * u.x + u.y * u.y)

# Cálculo do ângulo entre três pontos A, B, C (ângulo em B)
def angle_3points(a: Point2D, b: Point2D, c: Point2D,) -> Optional[float]:
    """
    Calcula o ângulo ABC (em graus), isto é, o ângulo no ponto B
    formado pelos segmentos BA e BC.

    Fórmula:
        u = A - B
        v = C - B
        cos(theta) = (u·v) / (||u|| ||v||)
        theta = arccos(cos(theta)) em graus

    Retorna:
        float (graus) ou None se não for possível (ex.: pontos coincidentes gerando vetor zero).
    """
    # Vetores u = BA e v = BC, mas usando A-B e C-B
    u = Point2D(a.x - b.x, a.y - b.y)
    v = Point2D(c.x - b.x, c.y - b.y)

    nu = _norm(u)
    nv = _norm(v)

    # Se algum vetor tem norma 0, não existe ângulo definido
    if nu == 0 or nv == 0:
        return None

    d = _dot(u, v)

    # cos(theta) = d / (nu*nv)
    denom = nu * nv
    cos_theta = d / denom

    # Corrige possíveis erros numéricos que deixem cos_theta fora do intervalo [-1, 1]
    if cos_theta > 1:
        cos_theta = 1.0
    elif cos_theta < -1:
        cos_theta = -1.0

    theta_rad = acos(cos_theta)
    theta_deg = degrees(theta_rad)

    return theta_deg


def angular_variation(angles: Iterable[float], *, debug: bool = False) -> float:
    """
    Retorna a variação angular: max(angles) - min(angles).

    Útil para:
      - amplitude do cotovelo
      - quanto o tronco balançou

    Se a lista estiver vazia, lança ValueError.
    """
    angles_list = list(angles)
    if not angles_list:
        raise ValueError("Lista de ângulos vazia. Não é possível calcular variação.")

    min_a = min(angles_list)
    max_a = max(angles_list)
    var = max_a - min_a

    return var