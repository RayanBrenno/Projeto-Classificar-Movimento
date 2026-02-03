from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional
import cv2
import mediapipe as mp
from angle_utils import Point2D

# Um frame = dicionário com landmarks importantes
LandmarkFrame = Dict[str, Optional[Point2D]]


@dataclass
class ExtractionConfig:
    side: str = "right"
    visibility_threshold: float = 0.5
    max_frames: Optional[int] = None
    step: int = 1
    model_complexity: int = 1
    smooth_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def _landmark_to_point2d(lm, *, visibility_threshold: float) -> Optional[Point2D]:
    """
    Converte landmark do MediaPipe (x,y,visibility) para Point2D(x,y).
    Retorna None se visibility < threshold.
    Obs: x,y são normalizados (0..1).
    """
    if lm is None:
        return None
    vis = getattr(lm, "visibility", 0.0)
    if vis is None:
        vis = 0.0
    if vis < visibility_threshold:
        return None
    return Point2D(float(lm.x), float(lm.y))


def extract_landmarks_from_video(*, video_path: str, side: str = "right", visibility_threshold: float = 0.5, save_csv: bool = False, csv_path: Optional[str] = None, save_annotated_video: bool = False, annotated_video_path: Optional[str] = None, config: Optional[ExtractionConfig] = None) -> List[LandmarkFrame]:
    """
    Lê vídeo, roda MediaPipe Pose, devolve frames no formato:
      {"shoulder": Point2D|None, "elbow": ..., "wrist": ..., "hip": ..., "knee": ...}

    Parâmetros:
      - side: "right" ou "left"
      - visibility_threshold: abaixo disso, landmark vira None
      - save_csv/csv_path: salva CSV com x,y e flag de detecção
      - save_annotated_video/annotated_video_path: salva vídeo com pose desenhada
    """
    if config is None:
        config = ExtractionConfig(
            side=side, visibility_threshold=visibility_threshold)

    side = side.lower().strip()
    if side not in ("right", "left"):
        raise ValueError("side precisa ser 'right' ou 'left'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Preparar writer do vídeo anotado (se quiser)
    out_writer = None
    if save_annotated_video:
        if not annotated_video_path:
            raise ValueError(
                "annotated_video_path é obrigatório quando save_annotated_video=True")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(
            annotated_video_path, fourcc, fps, (width, height))

    # Preparar CSV (se quiser)
    csv_file = None
    csv_writer = None
    if save_csv:
        if not csv_path:
            raise ValueError("csv_path é obrigatório quando save_csv=True")
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame",
            "pose_detected",
            "shoulder_x", "shoulder_y",
            "elbow_x", "elbow_y",
            "wrist_x", "wrist_y",
            "hip_x", "hip_y",
            "knee_x", "knee_y",
        ])

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    # Índices do MediaPipe
    # (pontos do lado escolhido)
    if side == "right":
        idx_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        idx_elbow = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        idx_wrist = mp_pose.PoseLandmark.RIGHT_WRIST.value
        idx_hip = mp_pose.PoseLandmark.RIGHT_HIP.value
        idx_knee = mp_pose.PoseLandmark.RIGHT_KNEE.value
    else:
        idx_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        idx_elbow = mp_pose.PoseLandmark.LEFT_ELBOW.value
        idx_wrist = mp_pose.PoseLandmark.LEFT_WRIST.value
        idx_hip = mp_pose.PoseLandmark.LEFT_HIP.value
        idx_knee = mp_pose.PoseLandmark.LEFT_KNEE.value

    frames_out: List[LandmarkFrame] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=config.model_complexity,
        smooth_landmarks=config.smooth_landmarks,
        enable_segmentation=False,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    ) as pose:

        frame_index = 0
        kept_count = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Pular frames (acelera)
            if config.step > 1 and (frame_index % config.step != 0):
                frame_index += 1
                continue

            # OpenCV -> RGB pro mediapipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            pose_detected = results.pose_landmarks is not None

            # Por padrão, retorna None se não detectou pose
            lm_shoulder = lm_elbow = lm_wrist = lm_hip = lm_knee = None

            if pose_detected:
                lms = results.pose_landmarks.landmark
                lm_shoulder = lms[idx_shoulder]
                lm_elbow = lms[idx_elbow]
                lm_wrist = lms[idx_wrist]
                lm_hip = lms[idx_hip]
                lm_knee = lms[idx_knee]

            p_shoulder = _landmark_to_point2d(
                lm_shoulder, visibility_threshold=config.visibility_threshold)
            p_elbow = _landmark_to_point2d(
                lm_elbow, visibility_threshold=config.visibility_threshold)
            p_wrist = _landmark_to_point2d(
                lm_wrist, visibility_threshold=config.visibility_threshold)
            p_hip = _landmark_to_point2d(
                lm_hip, visibility_threshold=config.visibility_threshold)
            p_knee = _landmark_to_point2d(
                lm_knee, visibility_threshold=config.visibility_threshold)

            frame_dict: LandmarkFrame = {
                "shoulder": p_shoulder,
                "elbow": p_elbow,
                "wrist": p_wrist,
                "hip": p_hip,
                "knee": p_knee,
            }
            frames_out.append(frame_dict)

            # CSV
            if csv_writer is not None:
                def xy(p: Optional[Point2D]):
                    return ("", "") if p is None else (p.x, p.y)

                sx, sy = xy(p_shoulder)
                ex, ey = xy(p_elbow)
                wx, wy = xy(p_wrist)
                hx, hy = xy(p_hip)
                kx, ky = xy(p_knee)

                csv_writer.writerow([
                    frame_index,
                    1 if pose_detected else 0,
                    sx, sy, ex, ey, wx, wy, hx, hy, kx, ky
                ])

            # Vídeo anotado
            if out_writer is not None:
                annotated = frame_bgr.copy()
                if pose_detected:
                    mp_draw.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )
                out_writer.write(annotated)

            kept_count += 1
            frame_index += 1

            if config.max_frames is not None and kept_count >= config.max_frames:
                break

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if csv_file is not None:
        csv_file.close()

    return frames_out
