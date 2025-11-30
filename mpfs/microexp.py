# mpfs/microexp.py
"""
Módulo de extração de features e detecção de microexpressões.

Fluxo geral:
- Recebe landmarks 2D em pixels (FaceMesh estabilizado via LandmarkStabilizer).
- Calcula deslocamentos, velocidades e métricas por ROI (boca, sobrancelha, olhos etc.).
- Calcula perímetro/área de ROIs e áreas de triângulos (strain).
- Mantém histórico por métrica para calcular z-score adaptativo.
- Detecta "micro-eventos" curtos (40–500 ms) quando uma métrica passa um limiar de z-score.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

from .roi import ROI, PAIRS, TRIS
from .stabilizer import LandmarkStabilizer


def area_triangle(a, b, c):
    """
    Calcula a área de um triângulo 2D definido por três pontos.

    Usa a fórmula baseada no produto vetorial:
        area = 0.5 * | (b - a) x (c - a) |

    Parâmetros
    ----------
    a, b, c : np.ndarray
        Pontos 2D (x, y) representados como vetores NumPy de shape (2,)
        ou (N, 2) compatíveis com operações de subtração/np.cross.

    Retorna
    -------
    float
        Área do triângulo formado por (a, b, c).
    """
    return 0.5 * abs(np.cross(b - a, c - a))


class MicroExpDetector:
    """
    Detector de microexpressões baseado em features geométricas por ROI.

    Responsabilidades principais:
    - Estabilizar a cabeça (pose) usando pontos estáveis via LandmarkStabilizer.
    - Calcular deslocamento, velocidade (e opcionalmente aceleração) por ponto.
    - Agregar métricas por ROI (deslocamento/velocidade medianos, perímetro, área).
    - Calcular medidas "AU-like" com distâncias entre pares de pontos (PAIRS).
    - Calcular áreas de triângulos (TRIS) para strain/efeitos de deformação local.
    - Manter um histórico adaptativo por métrica (deque) para média/σ.
    - Detectar micro-eventos curtos (z-score acima de limiar, duração entre
      min_event_ms e max_event_ms).

    Atributos principais
    --------------------
    fps : float
        FPS nominal esperado (usado para dimensionar buffers).
    buf : int
        Tamanho do buffer (número de frames) usado para estatísticas adaptativas.
    z_k : float
        Limiar de z-score para disparar início de evento.
    max_event : int
        Duração máxima de um evento em milissegundos.
    min_event : int
        Duração mínima de um evento em milissegundos.
    last_events : list[dict]
        Lista de eventos detectados no último frame processado.
    last_pts : np.ndarray | None
        Cópia dos últimos pontos estabilizados (shape (468, 2)).
    last_vel_mag : np.ndarray | None
        Magnitude de velocidade por ponto (shape (468,)) referente ao último frame.
    """

    def __init__(
        self,
        fps_hint:     float = 30.0,
        buf_seconds:  float = 1.0,
        z_k:          float = 3.0,
        max_event_ms: int   = 500,
        min_event_ms: int   = 40
    ):
        """
        Inicializa o detector de microexpressões.

        Parâmetros
        ----------
        fps_hint : float
            FPS nominal esperado do stream de vídeo. Usado para dimensionar
            o tamanho do buffer temporal (buf_seconds * fps_hint).
        buf_seconds : float
            Janela temporal, em segundos, para o cálculo adaptativo de média
            e desvio-padrão das métricas (histórico deslizante).
        z_k : float
            Limiar de z-score para considerar que uma métrica entrou em evento.
        max_event_ms : int
            Duração máxima (em milissegundos) de um micro-evento.
        min_event_ms : int
            Duração mínima (em milissegundos) de um micro-evento. Eventos mais
            curtos são ignorados como ruído.

        Notas
        -----
        - `metric_hist` é um dicionário de deques: métrica -> histórico de valores.
        - `active_events` guarda o estado de eventos em andamento por métrica.
        """
        self.fps          = fps_hint
        self.buf          = int(max(5, buf_seconds * fps_hint))
        self.z_k          = z_k
        self.max_event    = max_event_ms
        self.min_event    = min_event_ms
        self.last_events  = []  # eventos detectados no último frame
        self.last_pts     = None           # (468,2) alinhados/estabilizados, em px
        self.last_vel_mag = None           # (468,) magnitude de velocidade por ponto (px/s)

        self.stab = LandmarkStabilizer()
        self.prev_pts:  Optional[np.ndarray] = None
        self.prev_time: Optional[int] = None

        # buffers por métrica → deque para média/σ adaptativos
        self.metric_hist:   Dict[str, deque] = {}
        self.active_events: Dict[str, Dict] = {}  # roi.metric → {"t_on":..., "peak":...}
        self.last_events = []  # guarda eventos do último frame

    # --------- API principal ----------
    def process(self, t_ms: int, pts2d: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Processa um frame de landmarks 2D e retorna um dicionário de features.

        Fluxo interno:
        1. Converte a lista de pontos em array NumPy (P) de shape (N, 2).
        2. Exige pelo menos 468 pontos (face mesh completo).
        3. Estabiliza P com LandmarkStabilizer (removendo movimento global de cabeça).
        4. Calcula deslocamento (D) e velocidade (V) normalizados por escala da face.
        5. Para cada ROI em ROI:
           - deslocamento mediano (disp),
           - velocidade mediana (vel),
           - perímetro e área aproximados do polígono da ROI.
        6. Para cada par em PAIRS:
           - distância euclidiana entre dois pontos (features "AU-like").
        7. Para cada triângulo em TRIS:
           - área do triângulo como proxy de strain local.
        8. Atualiza históricos e executa o detector de picos (z-score adaptativo)
           em métricas-chave, sinalizando início/fim de micro-eventos.
        9. Adiciona flags binárias `evt_<metric>` às features para cada evento
           que foi identificado neste frame.
        10. Atualiza estado interno (`prev_pts`, `prev_time`, `last_events`, etc.).

        Parâmetros
        ----------
        t_ms : int
            Timestamp do frame em milissegundos.
        pts2d : list[tuple[float, float]]
            Lista de landmarks 2D em coordenadas de pixel [(x, y), ...].

        Retorna
        -------
        dict[str, float]
            Dicionário de features numéricas prontas para uso em ML ou análise.
            Pode incluir:
            - `<roi>_disp`, `<roi>_vel`, `<roi>_perim`, `<roi>_area`
            - `<pair_name>` (distâncias)
            - `<tri_name>_area` (áreas)
            - `evt_<metric>` = 1.0 para eventos detectados neste frame.

        Notas
        -----
        - Se for o primeiro frame (sem histórico), a função retorna `{}`.
        - `last_events` contém uma lista de eventos finalizados neste frame
          com atributos como metric, t_on, t_off, dur_ms, peak, zmax.
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
            self.prev_pts = P_stab.copy()
            self.prev_time = t_ms
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
            area  = float(0.5 * np.abs(np.sum(
                pts[:, 0] * np.roll(pts[:, 1], -1) -
                pts[:, 1] * np.roll(pts[:, 0], -1)
            )))
            features[f"{roi_name}_disp"]  = disp
            features[f"{roi_name}_vel"]   = vel
            features[f"{roi_name}_perim"] = perim * norm
            features[f"{roi_name}_area"]  = area  * (norm**2)

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
            "brow_l_vel", "brow_r_vel", "eye_l_vel", "eye_r_vel",
            "mouth_corners_disp", "lips_out_disp", "lips_in_disp",
            "mouth_open", "lip_press", "lipCornerL_height", "lipCornerR_height"
        ]
        # fallback: se alguma chave não existir (nome diferente), derivar do features
        # mapeamos alguns alias rápidos
        alias = {
            "mouth_corners_disp": "mouth_corners_disp",
            "lips_out_disp":      "lips_out_disp",
            "lips_in_disp":       "lips_in_disp",
            "eye_l_vel":          "eye_l_vel",
            "eye_r_vel":          "eye_r_vel",
            "brow_l_vel":         "brow_l_vel",
            "brow_r_vel":         "brow_r_vel",
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
            mu    = (sum(hist) / len(hist)) if hist else 0.0
            sigma = (np.std(hist) if len(hist) > 3 else 1e-6)
            z     = (val - mu) / (sigma + 1e-6)
            hist.append(val)

            # estado do evento
            ev = self.active_events.get(key)
            if ev is None and z >= self.z_k:
                # start
                self.active_events[key] = {"t_on": t_ms, "peak": val, "zmax": z}
            elif ev is not None:
                ev["peak"] = max(ev["peak"], val)
                ev["zmax"] = max(ev["zmax"], z)
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
        self.prev_pts = P_stab.copy()
        self.prev_time = t_ms
        # anexar um “_event” contador simples por classe para ML supervisionado opcional
        for e in events_now:
            features[f"evt_{e['metric']}"] = 1.0

        self.last_events = events_now

        return features
