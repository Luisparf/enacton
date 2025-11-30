# mpfs/overlay.py
"""
Funções de desenho de overlay para o FaceSensor / coleta de microexpressões.

Responsabilidades principais:
- Conversão de landmarks normalizados (0–1) em coordenadas de pixel.
- Desenho de IDs, pontos, íris e wireframes da face (mesh MediaPipe).
- Função principal `draw_micro_overlay`, usada como callback de overlay no FaceSensor.
"""

from __future__ import annotations
from typing import Iterable, Optional, Tuple, List
import cv2
import numpy as np
import mediapipe as mp

from .contracts import FaceFrame, Landmark

# cores
# base
COLOR_FACE          = (255, 255, 255)   # branco
COLOR_DOTS          = (  0, 255,   0)   # verde
COLOR_IRIS          = (  0, 255, 255)   # amarelo
COLOR_TEXT          = (  0, 255, 255)   # amarelo
COLOR_WARN          = (  0, 165, 255)   # laranja
COLOR_IDS           = (  0, 230,  20)

# facemesh – BGR (comentário em RGB hex)
COLOR_RIGHT_EYE     = ( 48,  48, 255)  # #FF3030
COLOR_RIGHT_EYEBROW = ( 48,  48, 255)  # #FF3030
COLOR_RIGHT_IRIS    = ( 48,  48, 255)  # #FF3030

COLOR_LEFT_EYE      = ( 48, 255,  48)  # #30FF30
COLOR_LEFT_EYEBROW  = ( 48, 255,  48)  # #30FF30
COLOR_LEFT_IRIS     = ( 48, 255,  48)  # #30FF30

COLOR_FACE_OVAL     = (160, 160, 160)  # #E0E0E0
COLOR_LIPS          = (224, 224, 224)  # #E0E0E0


# helpers
def _to_pts(ff: FaceFrame) -> List[Tuple[int, int]]:
    """
    Converte landmarks normalizados de um FaceFrame em coordenadas de pixel (x, y).

    Cada Landmark em `ff.landmarks` possui coordenadas (x, y, z) normalizadas em [0, 1]
    em relação à largura/altura do frame. Esta função multiplica pelas dimensões do
    frame e retorna uma lista de pares (x, y) inteiros.

    Parâmetros
    ----------
    ff : FaceFrame
        Frame de face contendo largura, altura e a lista de landmarks normalizados.

    Retorna
    -------
    list[tuple[int, int]]
        Lista de coordenadas (x, y) já em pixels, no mesmo sistema de coordenadas
        do frame BGR utilizado pelo OpenCV.
    """
    h, w = ff.h, ff.w
    return [(int(lm.x * w), int(lm.y * h)) for lm in ff.landmarks]

#################################################################################################################

def _draw_ids(img, pts: List[Tuple[int,int]], color=COLOR_IDS):
    """
    Desenha o índice numérico de cada ponto ao lado da sua coordenada.

    Útil para debug/inspeção visual da numeração dos landmarks (por exemplo,
    quando se quer mapear manualmente regiões de interesse no FaceMesh).

    Parâmetros
    ----------
    img : np.ndarray
        Frame BGR no qual o texto será desenhado (modificado in-place).
    pts : list[tuple[int, int]]
        Lista de pontos (x, y) em pixels.
    color : tuple[int, int, int]
        Cor BGR usada para o texto.
    """
    for i, (x, y) in enumerate(pts):
        cv2.putText(
            img,
            str(i),
            (x+2, y-2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.2,
            color,
            1,
            cv2.LINE_AA
        )

#################################################################################################################

def _draw_all_points(img, pts: List[Tuple[int,int]]):
    """
    Desenha todos os pontos da face como pequenos círculos.

    Parâmetros
    ----------
    img : np.ndarray
        Frame BGR no qual os pontos serão desenhados (modificado in-place).
    pts : list[tuple[int, int]]
        Lista de pontos (x, y) em pixels a serem desenhados.
    """
    for (x, y) in pts:
        cv2.circle(img, (x, y), 1, COLOR_DOTS, -1, lineType=cv2.LINE_AA)

#################################################################################################################

def draw_mesh_mediapipe(img, face_landmarks_proto, thickness: int = 1):
    """
    Desenha o wireframe do MediaPipe FaceMesh usando as estruturas oficiais
    de conexões (olhos, sobrancelhas, lábios, contorno) com cores diferenciadas.

    Esta função assume que `face_landmarks_proto` é um
    `NormalizedLandmarkList` do MediaPipe, tipicamente construído a partir
    de `FaceFrame.landmarks`.

    Parâmetros
    ----------
    img : np.ndarray
        Frame BGR onde o mesh será desenhado (modificado in-place).
    face_landmarks_proto : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        Lista de landmarks normalizados no formato protobuf do MediaPipe.
    thickness : int, opcional
        Espessura padrão dos traços (pode ser sobrescrita internamente para
        algumas regiões, como sobrancelhas).
    """
    du = mp.solutions.drawing_utils
    fm = mp.solutions.face_mesh

    def spec(color, t=1):
        return du.DrawingSpec(color=color, thickness=thickness, circle_radius=0)

    # Tesselation geral (se quiser manter)
    du.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=fm.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec(COLOR_FACE_OVAL),
    )

    # Face oval
    du.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=fm.FACEMESH_FACE_OVAL,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec(COLOR_FACE_OVAL),
    )

    # Lábios
    du.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=fm.FACEMESH_LIPS,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec(COLOR_LIPS),
    )

    # Olho direito + sobrancelha + íris (vermelho #FF3030)
    du.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=fm.FACEMESH_RIGHT_EYE,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec(COLOR_RIGHT_EYE),
    )
    du.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=fm.FACEMESH_RIGHT_EYEBROW,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec(COLOR_RIGHT_EYEBROW, t=3),
    )
    du.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=fm.FACEMESH_RIGHT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec(COLOR_RIGHT_IRIS),
    )

    # Olho esquerdo + sobrancelha + íris (verde #30FF30)
    du.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=fm.FACEMESH_LEFT_EYE,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec(COLOR_LEFT_EYE),
    )
    du.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=fm.FACEMESH_LEFT_EYEBROW,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec(COLOR_LEFT_EYEBROW),
    )
    du.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=fm.FACEMESH_LEFT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec(COLOR_LEFT_IRIS),
    )

#################################################################################################################

def draw_micro_overlay(
    frame,                      # BGR
    ff: FaceFrame,              # contém w, h, landmarks, iris
    *,
    det=None,                   # MicroExpDetector (opcional, para eventos)
    show_all:    bool = False,
    show_ids:    bool = False,
    show_events: bool = False,
    mesh_style: Optional[str] = None,   # "mp" | "tri" | "grid" (por ora tratamos "mp" e genérico)
) -> None:
    """
    Desenha um overlay leve focado em microexpressões em cima do frame BGR.

    Responsabilidades:
    - Desenhar a malha facial (mesh) no estilo escolhido (`mesh_style`).
    - Desenhar marcadores de íris (crosshair) para facilitar a leitura do olhar.
    - Opcionalmente desenhar todos os pontos e/ou seus índices.
    - Opcionalmente exibir textos de eventos de microexpressão, se um detector
      (`det`) for passado e estiver mantendo um atributo `last_events`.

    Parâmetros
    ----------
    frame : np.ndarray
        Frame BGR no qual o overlay será desenhado (modificado in-place).
    ff : FaceFrame
        Estrutura contendo dimensões do frame, landmarks normalizados e íris.
    det : MicroExpDetector | None, opcional
        Detector de microexpressões. Se `show_events=True` e `det` não for None,
        o overlay tenta ler `det.last_events` para exibir informações textuais.
    show_all : bool, opcional
        Se True, desenha todos os landmarks como pontos (verde).
    show_ids : bool, opcional
        Se True, escreve o índice numérico ao lado de cada landmark.
    show_events : bool, opcional
        Se True, exibe textos dos eventos retornados pelo detector.
    mesh_style : str | None, opcional
        Estilo do mesh:
        - "mp"  : usa `draw_mesh_mediapipe` (FaceMesh completo).
        - None  : fallback minimalista com contorno, bocas e olhos em polilinhas.
        - "tri" / "grid": reservado para estilos futuros (por ora tratamos como genérico).

    Notas
    -----
    - Se `ff.landmarks` estiver vazio, a função retorna sem desenhar nada.
    - O overlay é desenhado diretamente em `frame` (sem cópia) para ser leve.
    """
    if not ff.landmarks:
        return

    h, w = ff.h, ff.w
    overlay = frame  # desenhamos diretamente (leve + AA)
    pts = _to_pts(ff)

    # 1) Mesh
    if mesh_style == "mp":
        # converter para proto NormalizedLandmarkList (MediaPipe)
        from mediapipe.framework.formats import landmark_pb2
        lm_proto = landmark_pb2.NormalizedLandmarkList()
        lm_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=float(lm.x), y=float(lm.y), z=float(lm.z)
            ) for lm in ff.landmarks
        ])
        draw_mesh_mediapipe(overlay, lm_proto, thickness=1)
    else:
        # contorno + olhos + boca (traço fino) como fallback
        OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
        LIPS = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317]
        LEFT_EYE  = [33,160,158,133,153,144]
        RIGHT_EYE = [263,387,385,362,380,373]

        def poly(idxs):
            return np.array([pts[i] for i in idxs], dtype=np.int32).reshape(-1, 1, 2)

        cv2.polylines(overlay, [poly(OVAL)],      True, COLOR_FACE, 1, cv2.LINE_AA)
        cv2.polylines(overlay, [poly(LIPS)],      True, COLOR_FACE, 1, cv2.LINE_AA)
        cv2.polylines(overlay, [poly(LEFT_EYE)],  True, COLOR_FACE, 1, cv2.LINE_AA)
        cv2.polylines(overlay, [poly(RIGHT_EYE)], True, COLOR_FACE, 1, cv2.LINE_AA)

    # 2) Íris (crosshair suave)
    for group in (ff.iris_left or [], ff.iris_right or []):
        for lm in group:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(overlay, (cx, cy), 2, COLOR_IRIS, -1, lineType=cv2.LINE_AA)
            cv2.line(overlay, (cx-5, cy), (cx+5, cy), COLOR_IRIS, 1, lineType=cv2.LINE_AA)
            cv2.line(overlay, (cx, cy-5), (cx, cy+5), COLOR_IRIS, 1, lineType=cv2.LINE_AA)

    # 3) Todos os pontos?
    if show_all:
        _draw_all_points(overlay, pts)

    # 4) IDs?
    if show_ids:
        _draw_ids(overlay, pts)

    # 5) Eventos do detector (opcional)
    if show_events and det is not None:
        evs = getattr(det, "last_events", []) or []
        if evs:
            x, y = 8, 20
            for ev in evs[:8]:
                txt = f"{ev['name']}  {ev['dur_ms']}ms  p={ev.get('z', 0):.3f}"
                cv2.putText(
                    overlay,
                    txt,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_WARN,
                    2,
                    cv2.LINE_AA
                )
                y += 22
