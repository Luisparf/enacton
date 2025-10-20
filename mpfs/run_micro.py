# mpfs/run_micro.py
from __future__ import annotations
import argparse, sys, csv, time
import pandas as pd
from  collections import OrderedDict
from .face_sensor import FaceSensor
from .contracts import FaceFrame
from .microexp import MicroExpDetector
import numpy as np
import cv2
from .overlay import draw_micro_overlay
from .roi import ROI
import os
from .overlay import draw_micro_overlay, draw_full_landmarks, draw_iris

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m mpfs.run_micro",
        description="Coleta de microexpressões (features + eventos) com MediaPipe FaceMesh."
    )
    p.add_argument("--cam",      type=int,    default=0)
    p.add_argument("--width",    type=int,    default=640)
    p.add_argument("--height",   type=int,    default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--preview", action="store_true", help="Mostra overlay do FaceSensor.")
    p.add_argument("--no-preview", dest="preview", action="store_false")
    p.set_defaults(preview=True)
    p.add_argument("--csv",       type=str,   default="",  help="Salvar features em CSV.")
    p.add_argument("--parquet",   type=str,   default="",  help="Salvar features em Parquet.")
    p.add_argument("--z",         type=float, default=3.0, help="Limite z-score para evento (default 3.0).")
    p.add_argument("--buf",       type=float, default=1.0, help="Janela (s) para baseline adaptativo (default 1.0s).")
    p.add_argument("--max-event", type=int,   default=500)
    p.add_argument("--min-event", type=int,   default=40)
    p.add_argument("--show-all",  action="store_true",     help="Desenhar todos os 468 pontos.")
    p.add_argument("--show-ids",  action="store_true",     help="Mostrar IDs dos pontos (polui a tela).")
    p.add_argument("--no-roi",    action="store_true",     help="Não desenhar ROIs, só nuvem completa.")

    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    det = MicroExpDetector(fps_hint=args.fps, buf_seconds=args.buf, z_k=args.z,
                           max_event_ms=args.max_event, min_event_ms=args.min_event)

    sensor = FaceSensor(
                        cam_index=args.cam, 
                        width=args.width, 
                        height=args.height,
                        enable_preview=args.preview, 
                        overlay_style="contours",
                        line_thickness=1,
                        debug_indices=False, 
                        enable_emotion=False)

    rows = []
    last_feats = {}
    last_pts = []
    last_events = []

    t0 = time.time()

    def on_frame(ff: FaceFrame):
        nonlocal last_feats, last_pts, last_events

        if not ff.landmarks:
            return
        # montar pts2d normalizados para pixels (o FaceSensor trabalha em pixels)
        # ff.landmarks: List[Landmark(x,y,z,vis)]
        # x,y são normalizados (0..1) no FaceMesh? No FaceSensor convertíamos ao desenhar.
        # Aqui assumimos que ff.landmarks já são normalizados por frame => precisamos do tamanho.
        # Como estamos dentro do FaceSensor, podemos derivar pixels pelo preview? Em vez disso,
        # altere o FaceSensor para armazenar (w,h) atuais no próprio objeto ou passe via callback.
        # Para agora, vamos assumir foram convertidos para pixels antes de montar FaceFrame.
        pts_px = [(lm.x * ff.w, lm.y * ff.h) for lm in ff.landmarks]
        last_pts = pts_px

        feats = det.process(ff.t, pts_px)   # detector sempre em pixels
        if feats:
            last_feats  = feats
            last_events = det.last_events
            feats       = OrderedDict(sorted(feats.items()))
            feats["t"]  = ff.t
            rows.append(feats)
        else:
            return
        feats = OrderedDict(sorted(feats.items()))
        feats["t"] = ff.t

    def overlay_cb(frame_bgr, ff):
    # 1) ROIs/barras primeiro (faz o blend)
        if last_feats:
            draw_micro_overlay(frame_bgr, last_pts, last_feats, last_events, alpha=0.25)

        # 2) Nuvem completa + IDs por cima (não some no blend)
        if args.show_all and last_pts:
            draw_full_landmarks(
                frame_bgr,
                last_pts,
                speeds=None,              # cor fixa (verde)
                show_ids=args.show_ids,   # ← IDs por cima
                radius=1
            )





    sensor.on_frame(on_frame)     
    sensor.set_overlay(overlay_cb) 

    try:
        sensor.start()
    finally:
        if rows:
            df = pd.DataFrame(rows).fillna(0.0)
            if args.csv:
                os.makedirs("data/microoutputs", exist_ok=True)  # <-- ADD
                df.to_csv("data/microoutputs/"+args.csv, index=False)
                print(f"[run_micro] CSV salvo em: data/microoutputs/{args.csv} ({len(df)} linhas)")
            if args.parquet:
                df.to_parquet(args.parquet, index=False)
                print(f"[run_micro] Parquet salvo em: {args.parquet} ({len(df)} linhas)")


if __name__ == "__main__":
    main()
