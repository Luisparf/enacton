# mpfs/microexp.py
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from .roi import ROI, PAIRS, TRIS
from .stabilizer import LandmarkStabilizer

def area_triangle(a,b,c):
    return 0.5 * abs(np.cross(b-a, c-a))

class MicroExpDetector:
    """
    Extrai features “micro” frame-a-frame e detecta micro-eventos por ROI.
    - Estabiliza cabeça (semelhança) com pontos estáveis.
    - Calcula deslocamento/vel/acc por ROI.
    - Calcula métricas “AU-like” (aperturas/pressões/alturas).
    - Calcula strain (área triângulos) e var. de perímetro/área de ROI.
    - Detecção por picos (z-score adaptativo, curta duração).
    """
    def __init__(self,
                 fps_hint: float = 30.0,
                 buf_seconds: float = 1.0,
                 z_k: float = 3.0,
                 max_event_ms: int = 500,
                 min_event_ms: int = 40):
        self.fps = fps_hint
        self.buf = int(max(5, buf_seconds * fps_hint))
        self.z_k = z_k
        self.max_event = max_event_ms
        self.min_event = min_event_ms
        self.last_events = []  # eventos detectados no último frame
        self.last_pts = None           # (468,2) alinhados/estabilizados, em px
        self.last_vel_mag = None       # (468,) magnitude de velocidade por ponto (px/s)

        self.stab = LandmarkStabilizer()
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_time: Optional[int] = None

        # buffers por métrica → deque para média/σ adaptativos
        self.metric_hist: Dict[str, deque] = {}
        self.active_events: Dict[str, Dict] = {}  # roi.metric → {"t_on":..., "peak":...}
        

    # --------- API principal ----------
    def process(self, t_ms: int, pts2d: List[Tuple[float,float]]) -> Dict[str, float]:
        """
        pts2d: lista [(x,y)] com todos os landmarks (em pixels).
        Retorna dict de features (pronto para ML).
        """
        P = np.array(pts2d, dtype=np.float32)
        if P.shape[0] < 468:  # exigir face mesh completo
            return {}

        # 1) estabilizar
        P_stab, scale = self.stab.apply(P)

        # 2) normalizador métrico (altura da face já internalizado na estabilização)
        norm = 1.0  # já normalizado pela ref; ainda assim aplicamos /scale para segurança
        norm *= (1.0 / (scale + 1e-8))

        # 3) delta/vel/acc
        features: Dict[str, float] = {}
        if self.prev_pts is None or self.prev_time is None:
            self.prev_pts = P_stab.copy(); self.prev_time = t_ms
            return {}

        dt = max(1e-3, (t_ms - self.prev_time) / 1000.0)
        D = (P_stab - self.prev_pts) * norm
        V = D / dt
        self.last_pts = P_stab.copy()
        self.last_vel_mag = np.linalg.norm(V, axis=1)  # velocidade por ponto

        # não precisamos de A, mas mantemos caso use
        # A = (V - ((self.prev_pts - self.prev_prev) * norm / last_dt)) / dt

        # 4) por ROI — deslocamento, perímetro e área
        for roi_name, idxs in ROI.items():
            pts = P_stab[idxs]
            # deslocamento mediano dos pontos da ROI
            disp = np.median(np.linalg.norm(D[idxs], axis=1))
            vel  = np.median(np.linalg.norm(V[idxs], axis=1))
            # perímetro/área (polígono simples)
            perim = float(np.sum(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)))
            area  = float(0.5 * np.abs(np.sum(pts[:,0]*np.roll(pts[:,1],-1) - pts[:,1]*np.roll(pts[:,0],-1))))
            features[f"{roi_name}_disp"] = disp
            features[f"{roi_name}_vel"]  = vel
            features[f"{roi_name}_perim"]= perim * norm
            features[f"{roi_name}_area"] = area  * (norm**2)

        # 5) pares AU-like (distâncias verticais/relativas)
        for name, (i, j) in PAIRS.items():
            d = float(np.linalg.norm(P_stab[i] - P_stab[j])) * norm
            features[f"{name}"] = d

        # 6) triângulos (strain como Δárea relativa)
        for name, (i, j, k) in TRIS.items():
            a = area_triangle(P_stab[i], P_stab[j], P_stab[k]) * (norm**2)
            # armazenamos a área normalizada; strain pode ser derivada via Δ relativo
            features[f"{name}_area"] = a

        # 7) detecção de micro-evento (z-score adaptativo por métrica-chave)
        key_metrics = [
            "brow_l_vel","brow_r_vel","eye_l_vel","eye_r_vel",
            "mouth_corners_disp","lips_out_disp","lips_in_disp",
            "mouth_open","lip_press","lipCornerL_height","lipCornerR_height"
        ]
        # fallback: se alguma chave não existir (nome diferente), derivar do features
        # mapeamos alguns alias rápidos
        alias = {
            "mouth_corners_disp" :  "mouth_corners_disp",
            "lips_out_disp"      :  "lips_out_disp",
            "lips_in_disp"       :  "lips_in_disp",
            "eye_l_vel"          :  "eye_l_vel",
            "eye_r_vel"          :  "eye_r_vel",
            "brow_l_vel"         :  "brow_l_vel",
            "brow_r_vel"         :  "brow_r_vel",
        }

        # se alguma chave não existir, tente compor:
        if "mouth_corners_disp" not in features:
            # derive da ROI mouth_corners se existir
            if "mouth_corners_disp" not in features and "mouth_corners" in ROI:
                # já calculado acima como ROI genérico; se não, calcule rápido
                idxs = ROI["mouth_corners"]
                features["mouth_corners_disp"] = float(np.median(np.linalg.norm(D[idxs], axis=1)))

        # executar detector por métrica presente
        events_now = []
        for key in key_metrics:
            if key not in features:
                continue
            val   = float(features[key])
            hist  = self.metric_hist.setdefault(key, deque(maxlen=self.buf))
            mu    = (sum(hist)/len(hist)) if hist else 0.0
            sigma = (np.std(hist) if len(hist) > 3 else 1e-6)
            z     = (val - mu) / (sigma + 1e-6)
            hist.append(val)

            # estado do evento
            ev = self.active_events.get(key)
            if ev is None and z >= self.z_k:
                # start
                self.active_events[key] = {"t_on": t_ms, "peak": val, "zmax": z}
            elif ev is not None:
                ev["peak"] = max(ev["peak"], val); ev["zmax"] = max(ev["zmax"], z)
                dur = t_ms - ev["t_on"]
                # encerra se caiu abaixo de 1σ ou passou max_event
                if z < 1.0 or dur >= self.max_event:
                    if self.min_event <= dur <= self.max_event:
                        events_now.append({
                            "metric": key,
                            "t_on":   ev["t_on"],
                            "t_off":  t_ms,
                            "dur_ms": dur,
                            "peak":   ev["peak"],
                            "zmax":   ev["zmax"],
                        })
                    self.active_events.pop(key, None)

        # atualizar prev
        self.prev_pts = P_stab.copy(); self.prev_time = t_ms
        # anexar um “_event” contador simples por classe para ML supervisionado opcional
        for e in events_now:
            features[f"evt_{e['metric']}"] = 1.0

        self.last_events = events_now

        return features
