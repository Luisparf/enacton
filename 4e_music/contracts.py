# contracts.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import time

"""
Embodied → PoseSensor (OpenCV + MediaPipe) + GestureEncoder (heurístico agora; PyTorch depois)

Embedded → ContextProvider (objeto simples que injeta contexto correntemente)

Enactive → InteractionEngine (estado + política: gesto + contexto → ações)

Extended → ActuatorBus (envia OSC ou MIDI com python-osc/mido)

Camera → PoseSensor → GestureEncoder → InteractionEngine ⇄ ContextProvider → ActuatorBus → (synth/DAW/lights)


"""
@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0
    v: float = 1.0  # visibility/confidence opcional

@dataclass
class PoseFrame:
    t: float
    left_hand: List[Landmark]
    right_hand: List[Landmark]
    body: List[Landmark]

@dataclass
class GestureEvent:
    t: float
    features: np.ndarray          # shape [F]
    labels: Optional[List[str]] = None
    confidence: Optional[float] = None

@dataclass
class Context:
    t: float
    instrument: str = "guitar"
    stage_zone: Optional[str] = None
    piece_section: Optional[str] = None
    session_mode: str = "rehearsal"
    constraints: Optional[Dict[str, Any]] = None

@dataclass
class Action:
    t: float
    target: str                   # "synth" | "daw" | "lights" | "viz"
    msg_type: str                 # "OSC" | "MIDI"
    path: Optional[str]           # para OSC
    data: Any
    priority: int = 0

def now_ms() -> float:
    return time.time() * 1000.0
