import numpy as np
from .contracts import GestureEvent, FaceEvent, MultimodalEvent

def fuse(body: GestureEvent | None, face: FaceEvent | None) -> MultimodalEvent:
    parts = []
    if body is not None: parts.append(body.features)
    if face is not None: parts.append(face.features)
    feats = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
    t = (face.t if face else (body.t if body else 0.0))
    return MultimodalEvent(t=t, body=body, face=face, features=feats)
