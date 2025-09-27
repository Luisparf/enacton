import numpy as np
from .contracts import FaceFrame, FaceEvent


def _eye_aspect_ratio(eye_pts):
    """
    Calcula o Eye Aspect Ratio (EAR), métrica clássica para detectar piscadas.

    Args:
        eye_pts (List[Landmark]): landmarks de um olho (cantos e pálpebras).

    Returns:
        float: valor do EAR. Quanto menor, mais fechado está o olho.
    """
    if not eye_pts:
        return 0.0
    # TODO: implementar cálculo real usando pares de pálpebras.
    return 0.0


def _iris_center(iris_pts):
    """
    Calcula o centro da íris como média das coordenadas.

    Args:
        iris_pts (List[Landmark]): landmarks da íris.

    Returns:
        tuple(float, float): coordenadas (x, y) médias da íris.
    """
    if not iris_pts:
        return (0.0, 0.0)
    xs = np.array([[p.x, p.y] for p in iris_pts])
    c = xs.mean(0)
    return float(c[0]), float(c[1])


class HeuristicFaceEncoder:
    """
    Traduz um `FaceFrame` em um `FaceEvent` usando heurísticas simples.

    Responsabilidades:
        - Calcular probabilidade de piscada a partir do EAR.
        - Estimar direção do olhar (gaze) a partir do centro da íris.
        - Estimar abertura da boca (stub inicial).
        - Gerar vetor de features numéricas representando o rosto em um instante.

    Features atuais:
        [blink, gaze_dx, gaze_dy, mouth_open]

    Uso:
        encoder = HeuristicFaceEncoder()
        ff = FaceFrame(...)   # vindo do FaceSensor
        fe = encoder.encode(ff)
    """

    def __init__(self):
        pass

    def encode(self, ff: FaceFrame) -> FaceEvent:
        """
        Converte um `FaceFrame` em um `FaceEvent`.

        Args:
            ff (FaceFrame): frame facial bruto vindo do sensor.

        Returns:
            FaceEvent: evento processado com vetor de features.
        """
        # Blink: 1 - EAR (quanto menor EAR, mais fechado → blink alto)
        blink = 0.0  # TODO: implementar usando _eye_aspect_ratio

        # Gaze: deslocamento médio das duas íris
        gx_l, gy_l = _iris_center(ff.iris_left)
        gx_r, gy_r = _iris_center(ff.iris_right)
        gaze_dx = (gx_l + gx_r) / 2.0
        gaze_dy = (gy_l + gy_r) / 2.0

        # Abertura da boca (stub; precisa pares de landmarks lábios)
        mouth_open = 0.0

        feats = np.array([blink, gaze_dx, gaze_dy, mouth_open], dtype=np.float32)
        return FaceEvent(t=ff.t, features=feats)
