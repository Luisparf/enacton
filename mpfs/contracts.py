from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import time

"""
Embodied → PoseSensor (OpenCV + MediaPipe) + GestureEncoder (heurístico agora; PyTorch depois)

Embedded → ContextProvider (objeto simples que injeta contexto correntemente)

Enactive → InteractionEngine (estado + política: gesto + contexto → ações)

Extended → ActuatorBus (envia OSC ou MIDI com python-osc/mido)

Fluxo geral:
Camera → PoseSensor → GestureEncoder → InteractionEngine ⇄ ContextProvider → ActuatorBus → (synth/DAW/lights)

flowchart LR
    Camera --> PoseSensor
    PoseSensor --> PoseFrame
    PoseFrame --> GestureEncoder
    GestureEncoder --> GestureEvent
    GestureEvent --> InteractionEngine
    ContextProvider --> Context
    Context --> InteractionEngine
    InteractionEngine --> Action
    Action --> ActuatorBus
    ActuatorBus --> Synth[(Synth/DAW)]
    ActuatorBus --> Lights[(Luzes/Visuals)]

"""


@dataclass
class Landmark:
    """
    Representa um ponto do corpo (landmark) detectado pelo modelo de pose.

    Atributos:
        x (float): coordenada horizontal normalizada (0..1).
        y (float): coordenada vertical normalizada (0..1).
        z (float): profundidade relativa (0 = plano da câmera).
        v (float): visibilidade/confiança da detecção (0..1).
    """
    x: float
    y: float
    z: float 
    v: float = 1.0


@dataclass
class PoseFrame:
    """
    Snapshot do corpo em um instante de tempo.

    Atributos:
        t (float): timestamp em milissegundos.
        left_hand (List[Landmark]): landmarks da mão esquerda (ex.: 21 pontos do MediaPipe).
        right_hand (List[Landmark]): landmarks da mão direita.
        body (List[Landmark]): landmarks do corpo inteiro (ex.: 33 pontos do MediaPipe Pose).
    """
    t: float
    left_hand:  List[Landmark]
    right_hand: List[Landmark]
    body:       List[Landmark]


@dataclass
class GestureEvent:
    """
    Evento gestual processado a partir de uma janela de PoseFrames.

    Atributos:
        t (float): timestamp do evento (geralmente herdado do último frame).
        features (np.ndarray): vetor de características numéricas extraídas do gesto.
        labels (Optional[List[str]]): rótulos semânticos opcionais (ex.: "staccato").
        confidence (Optional[float]): nível de confiança na detecção/encoder.
    """
    t: float
    features:   np.ndarray
    labels:     Optional[List[str]] = None
    confidence: Optional[float] = None


@dataclass
class Context:
    """
    Contexto situacional da performance (informações que modulam interpretação).

    Atributos:
        t (float): timestamp em milissegundos.
        instrument (str): instrumento sendo tocado (ex.: "guitar").
        stage_zone (Optional[str]): posição no palco (ex.: "front", "left").
        piece_section (Optional[str]): seção da peça (ex.: "A", "B", "coda").
        session_mode (str): modo atual (ex.: "rehearsal" ou "performance").
        constraints (Optional[Dict[str, Any]]): restrições adicionais (ex.: limite de volume).
    """
    t: float
    instrument:    str = "guitar"
    stage_zone:    Optional[str] = None
    piece_section: Optional[str] = None
    session_mode:  str = "rehearsal"
    constraints:   Optional[Dict[str, Any]] = None


@dataclass
class Action:
    """
    Ação de saída gerada pelo sistema (mensagem para sintetizador, DAW, luzes, etc.).

    Atributos:
        t (float): timestamp em milissegundos.
        target (str): destino lógico da ação ("synth", "daw", "lights", "viz").
        msg_type (str): protocolo de saída ("OSC", "MIDI").
        path (Optional[str]): endereço da mensagem (ex.: "/engine" no OSC).
        data (Any): payload da mensagem (pode ser número, lista ou dict).
        priority (int): prioridade relativa da ação (0 = normal).
    """
    t: float
    target: str
    msg_type: str
    path: Optional[str]
    data: Any
    priority: int = 0


def now_ms() -> float:
    """
    Retorna o timestamp atual em milissegundos.

    Usado para manter consistência temporal em PoseFrame, GestureEvent, Context e Action.

    Returns:
        float: tempo atual em milissegundos desde época UNIX.
    """
    return time.time() * 1000.0


@dataclass
class FaceFrame:
    """
    Snapshot do rosto em um instante de tempo.

    Atributos:
        t (float): timestamp em milissegundos.
        landmarks (List[Landmark]): lista de ~468 landmarks do MediaPipe FaceMesh.
        iris_left (Optional[List[Landmark]]): landmarks da íris esquerda (~5 pontos).
        iris_right (Optional[List[Landmark]]): landmarks da íris direita (~5 pontos).
        gaze_vec (Optional[np.ndarray]): vetor normalizado (dx, dy) da direção do olhar.
        blink_prob (Optional[float]): probabilidade de piscar (0..1).
    """
    w: int
    h: int
    t: float
    landmarks:  List[Landmark]
    iris_left:  Optional[List[Landmark]]
    iris_right: Optional[List[Landmark]]
    gaze_vec:   Optional[np.ndarray] = None
    blink_prob: Optional[float] = None


@dataclass
class FaceEvent:
    """
    Evento processado a partir de uma janela de FaceFrames.

    Atributos:
        t (float): timestamp em milissegundos.
        features (np.ndarray): vetor de características faciais, ex.:
            [blink_rate, gaze_dx, gaze_dy, mouth_openness, ...].
        aus (Optional[Dict[str, float]]): intensidades de Action Units (AUs),
            ex.: {"AU01": 0.2, "AU12": 0.8}.
        expr (Optional[Dict[str, float]]): probabilidades de expressões básicas,
            ex.: {"happy": 0.6, "sad": 0.1}.
    """
    t: float
    features: np.ndarray
    aus: Optional[Dict[str, float]] = None
    expr: Optional[Dict[str, float]] = None
