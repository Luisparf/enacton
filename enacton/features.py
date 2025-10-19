"""
features.py
-----------
Registro central das features faciais: nomes, ordem, grupos e formatação.

Objetivo:
  - Centralizar a definição/ordem de atributos (evitar "caça" em múltiplos arquivos).
  - Padronizar como cada feature é exibida/impressa/salva (CSV/JSONL).
  - Permitir impressão agrupada (ex.: blink L/R/M em um bloco só).

Como usar:
  - Para ADICIONAR uma nova feature, edite APENAS `FEATURES_ORDER`
    (e, se quiser agrupar no console, edite `PRINT_GROUPS`).

Convenções:
  - Nomes curtos, snake_case.
  - Valores são floats (0..1 ou -0.5..+0.5 conforme o caso).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class FeatureSpec:
    """
    Especificação de uma feature.

    Atributos:
        name       : nome interno da feature (coluna CSV/JSON).
        fmt        : máscara de formatação para impressão.
        clip       : (min,max) para clamp opcional (pode ser ignorado pelo encoder).
        desc       : descrição curta (documentação humana).
    """
    name: str
    fmt: str = "{:.2f}"
    clip: Tuple[float, float] | None = None
    desc: str = ""


# Ordem canônica de gravação (CSV/JSONL) e leitura
FEATURES_ORDER: List[FeatureSpec] = [
    # Blink (L/R/M)
    FeatureSpec("blink_left",     "{:.2f}", (0.0, 1.0), "Piscada olho esquerdo"),
    FeatureSpec("blink_right",    "{:.2f}", (0.0, 1.0), "Piscada olho direito"),
    FeatureSpec("blink_mean",     "{:.2f}", (0.0, 1.0), "Piscada média"),

    # Gaze (direção média do olhar)
    FeatureSpec("gaze_dx",        "{:+.2f}", (-0.5, +0.5), "Olhar horizontal (médio)"),
    FeatureSpec("gaze_dy",        "{:+.2f}", (-0.5, +0.5), "Olhar vertical (médio)"),

    # Boca
    FeatureSpec("mouth_open",     "{:.2f}", (0.0, 1.0), "Abertura de boca"),

    # Sobrancelhas (L/R/M)
    FeatureSpec("brow_left",      "{:.2f}", (0.0, 1.0), "Sobrancelha esquerda (elevação)"),
    FeatureSpec("brow_right",     "{:.2f}", (0.0, 1.0), "Sobrancelha direita (elevação)"),
    FeatureSpec("brow_mean",      "{:.2f}", (0.0, 1.0), "Sobrancelhas média"),

    # Íris (raio normalizado, proxy pupilar) e confiança por olho
    FeatureSpec("iris_r_left",    "{:.2f}", (0.0, 1.0), "Raio íris (esq.) normalizado"),
    FeatureSpec("iris_r_right",   "{:.2f}", (0.0, 1.0), "Raio íris (dir.) normalizado"),
    FeatureSpec("iris_r_mean",    "{:.2f}", (0.0, 1.0), "Raio íris média"),

    FeatureSpec("gconf_left",     "{:.2f}", (0.0, 1.0), "Confiança gaze (esq.)"),
    FeatureSpec("gconf_right",    "{:.2f}", (0.0, 1.0), "Confiança gaze (dir.)"),
    FeatureSpec("gconf_mean",     "{:.2f}", (0.0, 1.0), "Confiança gaze média"),
]


# Grupos de impressão no console.
# Cada item: (rótulo, [nomes de features nessa ordem], formatação composta opcional)
PRINT_GROUPS: List[Tuple[str, List[str], Callable[[Dict[str, float]], str] | None]] = [
    ("gaze",  ["gaze_dx", "gaze_dy"], lambda d: f"({d['gaze_dx']:+.2f},{d['gaze_dy']:+.2f})"),
    ("mouth", ["mouth_open"],         None),

    # Ordem exigida pelo usuário: blink antes de brow
    ("blink", ["blink_left", "blink_right", "blink_mean"],
              lambda d: f"({d['blink_left']:.2f}/{d['blink_right']:.2f}/{d['blink_mean']:.2f})"),
    ("brow",  ["brow_left", "brow_right", "brow_mean"],
              lambda d: f"({d['brow_left']:.2f}/{d['brow_right']:.2f}/{d['brow_mean']:.2f})"),

    ("irisR", ["iris_r_left", "iris_r_right", "iris_r_mean"],
              lambda d: f"({d['iris_r_left']:.2f}/{d['iris_r_right']:.2f}/{d['iris_r_mean']:.2f})"),
    ("gconf", ["gconf_left", "gconf_right", "gconf_mean"],
              lambda d: f"({d['gconf_left']:.2f}/{d['gconf_right']:.2f}/{d['gconf_mean']:.2f})"),
]


# Helpers
def feature_names() -> List[str]:
    """Lista de nomes na ordem canônica."""
    return [f.name for f in FEATURES_ORDER]


def to_row(values: Dict[str, float]) -> List[str]:
    """Converte dicionário de valores → lista de strings (respeitando a ordem/format)."""
    out                      = []
    for spec in FEATURES_ORDER:
        v                    = float(values.get(spec.name, 0.0))
        out.append(spec.fmt.format(v))
    return out
