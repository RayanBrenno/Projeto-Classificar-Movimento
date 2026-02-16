# Classificação do Movimento – Remada Baixa

## Visão Geral

Sistema de classificação automática da execução da remada baixa usando visão computacional (MediaPipe Pose).

O projeto processa vídeos do exercício, extrai landmarks corporais quadro a quadro e calcula métricas biomecânicas focadas principalmente em cotovelo e tronco, gerando:

- Notas quantitativas (0 a 100)

- Classificação qualitativa (ok / médio / ruim)

- Feedbacks técnicos automáticos (alertas e sugestões)

A proposta é transformar a avaliação visual subjetiva do movimento em uma análise técnica baseada em dados, com uma pipeline modular e extensível para outros exercícios.

---

## Objetivo

Desenvolver um sistema capaz de avaliar a execução da remada baixa de forma objetiva, considerando:

- Amplitude do movimento do cotovelo

- Estabilidade durante a execução

- Controle do tronco

- Inclinação excessiva ou compensações

O objetivo principal é fornecer uma nota quantitativa e feedback técnico que auxiliem na correção da execução do exercício.

---

## Arquitetura do Projeto

O projeto foi estruturado de forma modular, separando responsabilidades para facilitar manutenção e evolução:

- Extração (MediaPipe Pose): leitura do vídeo + landmarks

- Séries temporais: transformação dos landmarks em séries relevantes

- Métricas biomecânicas: cálculo de ângulos, variações e estabilidade

- Scoring: geração de nota e feedback com base em faixas ideais

- Saídas: CSV + vídeo anotado + resultado final

Essa organização permite adaptar o pipeline para outros movimentos no futuro (ex: puxada, supino, agachamento).

---

## Requisitos Fundamentais

- Python 3.11 https://www.python.org/downloads/windows/
- mediapipe 0.10.14 (pip install -r requirements.txt)

Recomenda-se utilizar ambiente virtual (`venv`) para evitar conflitos de dependência.

---

## Criando o ambiente virtual (venv)

No terminal, dentro da pasta do projeto (raiz):

- Criar o ambiente virtual -> py -3.11 -m venv .venv

- Ativar o ambiente -> .venv\Scripts\activate

- Executar o projeto -> python src/main.py

---

## Como Funciona

O sistema segue o seguinte fluxo:

1. O vídeo da remada baixa é processado pelo MediaPipe Pose.
2. Os landmarks corporais são extraídos quadro a quadro.
3. São calculadas séries temporais dos pontos relevantes (ombro, cotovelo, quadril, etc.).
4. Métricas biomecânicas são derivadas, como:
   - Amplitude angular do cotovelo
   - Variação da inclinação do tronco
   - Estabilidade ao longo do movimento
5. Um sistema de pontuação compara os valores com faixas ideais pré-definidas.
6. São geradas:
   - Nota para cotovelo
   - Nota para tronco
   - Classificação qualitativa
   - Alertas técnicos, quando necessário
   - Vídeo processado com os pontos corporais mapeados
   - Arquivo CSV com as coordenadas normalizadas (x, y) dos landmarks ao longo do tempo

O resultado final é uma avaliação automatizada da execução da remada baixa, baseada em métricas objetivas.

---

## Saídas Geradas

O pipeline pode produzir:

- outputs/*.csv → coordenadas normalizadas (x, y) por frame

- outputs/*.mp4 → vídeo anotado com landmarks

- logs/prints → notas e feedback do moviment
