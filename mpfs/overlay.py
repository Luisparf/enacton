# mpfs/overlay.py
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, List, Tuple
from .roi import ROI

# Paleta simples
COL_ROI     = (255, 255, 255)
COL_HOT     = (0, 255, 255)
COL_TEXT    = (240, 240, 240)
COL_BAR     = (200, 200, 255)
COL_EVT     = (0, 200, 255)
HEAT_GAIN   = 10.0   # era 50.0
BAR_GAIN    = 10.0   # era 50.0
SHOW_LABELS = True
KEY_METRICS_ORDER = [
    "brow_l_vel","brow_r_vel","eye_l_vel","eye_r_vel",
    "mouth_open","lip_press","lipCornerL_height","lipCornerR_height",
    "lips_out_disp","lips_in_disp","mouth_corners_disp"
]
POINT_COLOR = (0, 255, 0)   # verde BGR
POINT_RADIUS = 1
# cores fixas
COL_ROI  = (255, 255, 255)   # branco
COL_TEXT = (240, 240, 240)
COL_EVT  = (0,   200, 255)

def _poly(frame, pts, color, thickness=1, closed=True):
    if len(pts) < 2: return
    ps = np.array(pts, dtype=np.int32).reshape(-1,1,2)
    cv2.polylines(frame, [ps], closed, color, thickness, cv2.LINE_AA)



def draw_micro_overlay(frame_bgr: np.ndarray,
                       pts2d: List[Tuple[float, float]],
                       feats: Dict[str, float],
                       events_now: List[Dict],
                       alpha: float = 0.25) -> None:
    """
    Desenha apenas os contornos/ROIs em branco (espessura 1) e,
    opcionalmente, barras/toasts (se você mantiver em outra parte).
    """
    overlay = frame_bgr.copy()

    # 1) ROIs (contornos brancos, finos)
    for roi_name, idxs in ROI.items():
        pts = [tuple(map(int, pts2d[i])) for i in idxs if i < len(pts2d)]
        if len(pts) >= 2:
            ps = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [ps], True, COL_ROI, 1, cv2.LINE_AA)  # espessura = 1

    # 2) Blend leve do overlay com a imagem base
    cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0, frame_bgr)

    # 3) Toasts de eventos
    if events_now:
        y_text = 24
        for ev in events_now[:6]:  # limita na tela
            msg = f"{ev['metric']} ↑ {ev['dur_ms']}ms p={ev['peak']:.3f}"
            cv2.putText(frame_bgr, msg, (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_EVT, 2, cv2.LINE_AA)
            y_text += 18


def _to_int_pts(pts):
    return [(int(x), int(y)) for x, y in pts]

def _color_from_scalar(v, vmin, vmax):
    # mapeia v -> cor usando colormap “JET” simples (BGR), clamp
    v = max(vmin, min(vmax, v))
    t = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)
    # JET approx
    r = int(255 * np.clip(1.5 - abs(2*t - 1), 0, 1))
    g = int(255 * np.clip(1.5 - abs(2*t - 0.5), 0, 1))
    b = int(255 * np.clip(1.5 - abs(2*t - 0.0), 0, 1))
    return (b, g, r)

def draw_full_landmarks(frame_bgr, pts_px, speeds=None, show_ids=False, radius=1, vmax_hint=None):
    # pontos (sempre verdes)
    for (x, y) in ((int(x), int(y)) for x, y in pts_px):
        cv2.circle(frame_bgr, (x, y), radius, POINT_COLOR, -1, cv2.LINE_AA)

    if not show_ids:
        return

    # IDs com contorno (preto) + preenchimento (branco) para legibilidade
    for i, (x, y) in enumerate((int(x), int(y)) for x, y in pts_px):
        org = (x + 2, y - 2)
        cv2.putText(frame_bgr, str(i), org, cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0),       2, cv2.LINE_AA)     # stroke
        cv2.putText(frame_bgr, str(i), org, cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_AA)  # fill

def draw_iris(frame_bgr, pts_px, color=(0,255,255)):
    """
    Se sua lista tem 478 pontos (ou você tem os índices de íris), desenha os 10 pontos de íris.
    """
    if len(pts_px) >= 478:
        IRIS_R = list(range(468, 473))
        IRIS_L = list(range(473, 478))
        for i in IRIS_R + IRIS_L:
            x, y = map(int, pts_px[i])
            cv2.circle(frame_bgr, (x,y), 2, color, -1, cv2.LINE_AA)
