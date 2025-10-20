# mpfs/roi.py
from __future__ import annotations

# Conjuntos de pontos do MediaPipe FaceMesh (468) – versão canônica
# Fonte cruzada: docs + prática. Mantidos curtos para serem estáveis/“grudados”.

# Face oval (contorno principal)
OVAL       = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
            152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

# Olhos (anel enxuto de 6 pontos)
LEFT_EYE   = [33,160,158,133,153,144]
RIGHT_EYE  = [263,387,385,362,380,373]

# Sobrancelhas (linha de 5 pontos)
LEFT_BROW  = [70,63,105,66,107]
RIGHT_BROW = [336,296,334,293,300]

# Lábios – borda externa compacta (não usa o anel completo pra evitar “sangrar”)
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317]

# Cantos de boca (apenas os extremos)
MOUTH_CORNERS = [61, 291]

# (opcional) Nariz BRIDGE curto (pra gente não desenhar aquele triângulo esquisito)
NOSE_BRIDGE = [168, 6, 197]  # centro/glabella – versão só pra métricas, NÃO desenhar

# Dicionário de ROIs que serão desenhadas
ROI = {
    "oval": OVAL,
    "eye_l": LEFT_EYE,
    "eye_r": RIGHT_EYE,
    "brow_l": LEFT_BROW,
    "brow_r": RIGHT_BROW,
    "lips_out": LIPS_OUTER,
    "mouth_corners": MOUTH_CORNERS,
    # "nose_bridge": NOSE_BRIDGE,  # -> deixamos fora do overlay pra não poluir
}

# Métricas “AU-like” por pares (distâncias/alturas)
PAIRS = {
    # Abertura de boca (13=upper inner, 14=lower inner)
    "mouth_open": (13, 14),
    # Pressão labial (distância horizontal dos cantos invertida – você pode inverter depois)
    "lip_press": (61, 291),
    # Altura canto da boca vs base lábio (proxy de AU12/15 – heurístico)
    "lipCornerL_height": (61, 146),
    "lipCornerR_height": (291, 375),
}

# Triângulos para strain local (bem conservadores)
TRIS = {
    "brow_l_tri": (70, 63, 105),
    "brow_r_tri": (336, 296, 334),
    "lip_tri": (61, 13, 291),
}
