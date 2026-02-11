# Classificação do Movimento – Remada Baixa

## Descrição

Sistema de classificação automática do movimento de remada baixa utilizando visão computacional com MediaPipe Pose.

O projeto analisa vídeos do exercício, extrai landmarks corporais e calcula métricas biomecânicas relacionadas principalmente ao cotovelo e ao tronco, gerando notas e feedbacks automáticos sobre a execução do movimento.

A proposta é transformar a avaliação visual subjetiva da remada baixa em uma análise técnica baseada em dados.

---

## Objetivo

Desenvolver um sistema capaz de avaliar a execução da remada baixa de forma objetiva, considerando:

- Amplitude do movimento do cotovelo
- Estabilidade durante a execução
- Controle do tronco
- Inclinação excessiva ou compensações

O objetivo principal é fornecer uma nota quantitativa e feedback técnico que auxiliem na correção da execução do exercício.

---

## Requisitos Fundamentais

- Python 3.11 https://www.python.org/downloads/windows/
- mediapipe 0.10.14 (pip install -r requirements.txt)

Recomenda-se utilizar ambiente virtual (`venv`) para evitar conflitos de dependência.

---

## Criando o ambiente virtual (venv)

No terminal, dentro da pasta do projeto (raiz):

Criar o ambiente virtual -> py -3.11 -m venv .venv

Ativar o ambiente -> .venv\Scripts\activate

Executar o projeto -> python src/main.py

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
