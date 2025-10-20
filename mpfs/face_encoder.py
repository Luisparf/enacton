"""
face_encoder.py
---------------
Encoder heurístico: FaceFrame -> dict[str, float] (features nomeadas).

Agora retornamos um `FaceEvent` cuja `features` é um vetor NumPy,
mas é montado a partir de um dicionário conforme o registro em `features.py`.
"""

from typing import List, Tuple
import numpy as np
from .contracts       import FaceFrame, FaceEvent, Landmark
from .features        import feature_names

# Índices FaceMesh (subconjuntos)
LEFT_EYE_IDX             = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX            = [263, 387, 385, 362, 380, 373]
LEFT_BROW_IDX            = [70, 63, 105, 66, 107]
RIGHT_BROW_IDX           = [336, 296, 334, 293, 300]
UPPER_LIP_IDX            = 13
LOWER_LIP_IDX            = 14


def _points(face_lms: List[Landmark], idxs: List[int]) -> np.ndarray:
    if not face_lms or any(i >= len(face_lms) for i in idxs):
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([[face_lms[i].x, face_lms[i].y] for i in idxs], dtype=np.float32)

# ===================================================================================================

def _eye_aspect_ratio(eye_pts_xy: np.ndarray) -> float:
    if eye_pts_xy.shape != (6, 2):
        return 0.0
    p                        = eye_pts_xy
    import numpy.linalg as LA
    A                        = LA.norm(p[1] - p[5])
    B                        = LA.norm(p[2] - p[4])
    C                        = LA.norm(p[0] - p[3])
    ear                      = (A + B) / (2.0 * C + 1e-6)
    return float(np.clip(ear, 0.0, 1.0))

# ===================================================================================================

def _rect_from_eye(eye_pts_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if eye_pts_xy.size == 0:
        return np.array([0.0, 0.0]), np.array([1.0, 1.0])
    mn                       = eye_pts_xy.min(axis=0)
    mx                       = eye_pts_xy.max(axis=0)
    return mn, mx

# ===================================================================================================


def _iris_center(iris_pts: List[Landmark]) -> Tuple[float, float]:
    if not iris_pts:
        return 0.5, 0.5
    c                        = np.array([[p.x, p.y] for p in iris_pts], dtype=np.float32).mean(axis=0)
    return float(c[0]), float(c[1])

# ===================================================================================================

def _iris_center_radius_conf(iris_pts: List[Landmark]) -> Tuple[float, float, float]:
    if not iris_pts:
        return 0.5, 0.5, 0.0
    pts                      = np.array([[p.x, p.y] for p in iris_pts], dtype=np.float32)
    c                        = pts.mean(axis=0)
    dists                    = np.linalg.norm(pts - c, axis=1)
    r_mean                   = float(dists.mean())
    r_std                    = float(dists.std() + 1e-8)
    circ_conf                = float(np.clip(1.0 - (r_std / (r_mean + 1e-8)), 0.0, 1.0))
    r_conf                   = r_mean * circ_conf
    return float(c[0]), float(c[1]), float(r_conf)

# ===================================================================================================


def _mouth_open(face_lms: List[Landmark]) -> float:
    if not face_lms or max(UPPER_LIP_IDX, LOWER_LIP_IDX) >= len(face_lms):
        return 0.0
    dy                       = abs(face_lms[UPPER_LIP_IDX].y - face_lms[LOWER_LIP_IDX].y)
    return float(np.clip(dy * 5.0, 0.0, 1.0))

# ===================================================================================================

def _brow_raise_side(face_lms: List[Landmark], eye_idx: List[int], brow_idx: List[int]) -> float:
    eye_xy                  = _points(face_lms, eye_idx)
    brow_xy                 = _points(face_lms, brow_idx)
    if eye_xy.size == 0 or brow_xy.size == 0:
        return 0.0
    eye_mean_y              = float(eye_xy[:, 1].mean())
    brow_mean_y             = float(brow_xy[:, 1].mean())
    eye_width               = float(np.linalg.norm(eye_xy[0] - eye_xy[3])) + 1e-6
    dy_norm                 = (eye_mean_y - brow_mean_y) / eye_width
    return float(np.clip(dy_norm * 1.5, 0.0, 1.0))


class HeuristicFaceEncoder:
    """
    Converte um `FaceFrame` em `FaceEvent` com features nomeadas.
    Adição de novas features requer apenas:
      - gerá-las aqui no dict `vals`
      - e listá-las (nome/ordem) em `FEATURES_ORDER` (features.py).
    """

    def encode(self, ff: FaceFrame) -> FaceEvent:
        # Olhos
        l_eye                 = _points(ff.landmarks, LEFT_EYE_IDX)
        r_eye                 = _points(ff.landmarks, RIGHT_EYE_IDX)

        # Blink L/R/M
        ear_l                 = _eye_aspect_ratio(l_eye)
        ear_r                 = _eye_aspect_ratio(r_eye)
        blink_left            = float(np.clip(1.0 - 3.0 * ear_l, 0.0, 1.0))
        blink_right           = float(np.clip(1.0 - 3.0 * ear_r, 0.0, 1.0))
        blink_mean            = float((blink_left + blink_right) * 0.5)

        # Gaze médio
        gx_l, gy_l            = _iris_center(ff.iris_left)   if ff.iris_left  else (0.5, 0.5)
        gx_r, gy_r            = _iris_center(ff.iris_right)  if ff.iris_right else (0.5, 0.5)
        mn_l, mx_l            = _rect_from_eye(l_eye);  sz_l = (mx_l - mn_l + 1e-6)
        mn_r, mx_r            = _rect_from_eye(r_eye);  sz_r = (mx_r - mn_r + 1e-6)
        rx_l                  = (gx_l - mn_l[0]) / sz_l[0];  ry_l = (gy_l - mn_l[1]) / sz_l[1]
        rx_r                  = (gx_r - mn_r[0]) / sz_r[0];  ry_r = (gy_r - mn_r[1]) / sz_r[1]
        gaze_dx               = float(np.clip(((rx_l + rx_r) / 2.0) - 0.5, -0.5, 0.5))
        gaze_dy               = float(np.clip(((ry_l + ry_r) / 2.0) - 0.5, -0.5, 0.5))

        # Boca
        mouth_open            = _mouth_open(ff.landmarks)

        # Sobrancelhas
        brow_left             = _brow_raise_side(ff.landmarks, LEFT_EYE_IDX,  LEFT_BROW_IDX)
        brow_right            = _brow_raise_side(ff.landmarks, RIGHT_EYE_IDX, RIGHT_BROW_IDX)
        brow_mean             = float((brow_left + brow_right) * 0.5)
    # ===================================================================================================

        # Íris (raio/“confiança” normalizados por olho)
        def eye_width(px: np.ndarray) -> float:
            if px.shape != (6, 2):
                return 1.0
            return float(np.linalg.norm(px[0] - px[3])) + 1e-6

        _, _, rconf_l         = _iris_center_radius_conf(ff.iris_left)   if ff.iris_left  else (0.5, 0.5, 0.0)
        _, _, rconf_r         = _iris_center_radius_conf(ff.iris_right)  if ff.iris_right else (0.5, 0.5, 0.0)
        w_l                   = eye_width(l_eye)
        w_r                   = eye_width(r_eye)

        iris_r_left           = float(np.clip(rconf_l / w_l, 0.0, 1.0))
        iris_r_right          = float(np.clip(rconf_r / w_r, 0.0, 1.0))
        iris_r_mean           = float((iris_r_left + iris_r_right) * 0.5)

        gconf_left            = float(1.0 if (ff.iris_left  and rconf_l > 0) else 0.0)
        gconf_right           = float(1.0 if (ff.iris_right and rconf_r > 0) else 0.0)
        gconf_mean            = float((gconf_left + gconf_right) * 0.5)

        # --- Dict de valores (fonte da verdade) ---
        vals                  = {
            "blink_left"   : blink_left,
            "blink_right"  : blink_right,
            "blink_mean"   : blink_mean,

            "gaze_dx"      : gaze_dx,
            "gaze_dy"      : gaze_dy,

            "mouth_open"   : mouth_open,

            "brow_left"    : brow_left,
            "brow_right"   : brow_right,
            "brow_mean"    : brow_mean,

            "iris_r_left"  : iris_r_left,
            "iris_r_right" : iris_r_right,
            "iris_r_mean"  : iris_r_mean,

            "gconf_left"   : gconf_left,
            "gconf_right"  : gconf_right,
            "gconf_mean"   : gconf_mean,
        }

        # Materializa vetor na ORDEM do registro central
        ordered               = np.array([vals.get(name, 0.0) for name in feature_names()], dtype=np.float32)
        return FaceEvent(t=ff.t, features=ordered)
