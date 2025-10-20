"""
face_recorder.py
----------------
Grava as features faciais (ordem e formatação definidas em `features.py`).

Arquivos gerados:
  - CSV    : cabeçalho = ["t_ms"] + FEATURES_ORDER
  - JSONL  : dicionário {nome: valor} com as mesmas chaves
"""

import csv, json, time
from pathlib             import Path
from .contracts          import FaceEvent
from .features           import feature_names


class FaceRecorder:
    """
    Registra séries temporais de features faciais em CSV/JSONL.
    """
    def __init__(self, out_dir: str = "data/face"):
        self.out               = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        ts                     = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path          = self.out / f"face_{ts}.csv"
        self.jsonl_path        = self.out / f"face_{ts}.jsonl"

        self._csv              = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer           = csv.writer(self._csv)

        header                 = ["t_ms"] + feature_names()
        self._writer.writerow(header)

    def write(self, ev: FaceEvent):
        feats                  = ev.features.tolist()
        row                    = [f"{ev.t:.3f}"] + [f"{x:.6f}" for x in feats]
        self._writer.writerow(row)
        self._csv.flush()

        # JSONL com chaves explícitas (mesma ordem) — útil para parsing posterior
        obj                    = {"t": float(ev.t)}
        for name, val in zip(feature_names(), feats):
            obj[name]          = float(val)
        with self.jsonl_path.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self):
        try:
            self._csv.close()
        except Exception:
            pass
