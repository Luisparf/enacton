# mpfs/run_micro.py
"""
Módulo de coleta de microexpressões faciais usando MediaPipe FaceMesh.

Responsabilidades principais:
- Ler parâmetros de linha de comando (câmera, FPS, resolução, formato de saída etc.).
- Construir o sensor de face (FaceSensor) e o detector de microexpressões (MicroExpDetector).
- Registrar callbacks de extração de features e desenho de overlay.
- Executar o loop principal de captura e, ao final, persistir as features em CSV/Parquet.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import List, Tuple, Optional

import pandas as pd

from .contracts   import FaceFrame
from .face_sensor import FaceSensor
from .microexp    import MicroExpDetector
from .overlay     import draw_micro_overlay


def parse_args(argv=None) -> argparse.Namespace:
    """
    Faz o parsing dos argumentos de linha de comando para o script de coleta.

    Parâmetros
    ----------
    argv : list[str] | None
        Lista de argumentos a serem parseados. Se None, usa sys.argv.

    Retorna
    -------
    argparse.Namespace
        Objeto com todos os parâmetros configurados:
        - cam, width, height, fps
        - preview / no-preview
        - csv, parquet
        - z, buf, max-event, min-event
        - show-all, show-ids, mesh, events
    """
    p = argparse.ArgumentParser(
        prog="python -m mpfs.run_micro",
        description="Coleta de microexpressões (features + overlay) com MediaPipe FaceMesh."
    )

    # Câmera e preview
    p.add_argument("--cam",     type=int, default=0)
    p.add_argument("--width",   type=int, default=640)
    p.add_argument("--height",  type=int, default=480)
    p.add_argument("--fps",     type=int, default=30)
    p.add_argument("--preview", action="store_true", help="Mostra overlay/preview.")
    p.add_argument("--no-preview", dest="preview", action="store_false")
    p.set_defaults(preview=True)

    # Persistência
    p.add_argument("--csv",       type=str,   default="",  help="Salvar features em CSV (caminho).")
    p.add_argument("--parquet",   type=str,   default="",  help="Salvar features em Parquet (caminho).")

    # Detector
    p.add_argument("--z",         type=float, default=3.0, help="Limite z-score para evento.")
    p.add_argument("--buf",       type=float, default=1.0, help="Janela (s) para baseline adaptativo.")
    p.add_argument("--max-event", type=int,   default=500, help="Duração máxima (ms) de um evento.")
    p.add_argument("--min-event", type=int,   default=40,  help="Duração mínima (ms) de um evento.")

    # Overlay e debug visual
    p.add_argument("--show-all", action="store_true", help="Desenha todos os pontos/landmarks.")
    p.add_argument("--show-ids", action="store_true", help="Escreve o índice em cada ponto (lento).")
    p.add_argument(
        "--mesh",
        choices=["tri", "grid", "mp"],
        default=None,
        help="Wireframe: 'tri' (tesselation), 'grid' (quadrícula), 'mp' (tesselation+contours do MediaPipe)."
    )
    p.add_argument(
        "--events",
        action="store_true",
        help="Mostra textos dos eventos detectados (por padrão, NÃO mostra)."
    )

    return p.parse_args(argv)

#################################################################################################################


def build_detector(args: argparse.Namespace) -> MicroExpDetector:
    """
    Constrói e configura um MicroExpDetector a partir dos argumentos de linha de comando.

    Parâmetros
    ----------
    args : argparse.Namespace
        Namespace contendo, no mínimo:
        - fps     : FPS nominal da captura
        - buf     : tamanho da janela (segundos) para baseline adaptativo
        - z       : limiar de z-score para sinalizar microexpressão
        - max_event, min_event : limites de duração do evento, em milissegundos

    Retorna
    -------
    MicroExpDetector
        Instância configurada do detector de microexpressões.
    """
    return MicroExpDetector(
        fps_hint=args.fps,
        buf_seconds=args.buf,
        z_k=args.z,
        max_event_ms=args.max_event,
        min_event_ms=args.min_event,
    )

#################################################################################################################


def build_sensor(args: argparse.Namespace) -> FaceSensor:
    """
    Constrói e configura o FaceSensor responsável pela captura de vídeo e FaceMesh.

    Parâmetros
    ----------
    args : argparse.Namespace
        Namespace com parâmetros de câmera e visualização:
        - cam, width, height, fps
        - preview
        - mesh: 'tri', 'grid', 'mp' (influencia o estilo de overlay)

    Retorna
    -------
    FaceSensor
        Instância configurada do sensor de face pronta para receber callbacks
        de processamento (on_frame) e overlay (set_overlay).
    """
    # novo estilo "mp": desenha tesselation + contornos (ver patch do FaceSensor abaixo)
    overlay_style = "mp" if args.mesh == "mp" else "hud"
    return FaceSensor(
        cam_index=args.cam,
        width=args.width,
        height=args.height,
        enable_preview=args.preview,
        debug_indices=False,
        enable_emotion=False,
        overlay_style=overlay_style,
        line_thickness=1,
        overlay_alpha=1.0,
    )

#################################################################################################################


def make_feature_callback(det: MicroExpDetector, rows_sink: List[OrderedDict]):
    """
    Cria um callback para ser registrado no FaceSensor e extrair features de cada frame.

    O callback resultante:
    - ignora frames sem landmarks;
    - converte landmarks normalizados (0–1) em coordenadas de pixel;
    - chama det.process(t, pts) para extrair features;
    - empacota as features (OrderedDict, ordenado por chave) e adiciona o timestamp 't';
    - armazena o resultado em rows_sink.

    Parâmetros
    ----------
    det : MicroExpDetector
        Detector de microexpressões responsável por processar as coordenadas dos pontos.
    rows_sink : list[OrderedDict]
        Lista mutável que servirá de “sink”/acumulador de linhas de features ao longo da captura.

    Retorna
    -------
    Callable[[FaceFrame], None]
        Função `on_frame(ff: FaceFrame) -> None` adequada para ser registrada em `sensor.on_frame`.
    """
    def on_frame(ff: FaceFrame) -> None:
        if not ff.landmarks:
            return
        pts: List[Tuple[float, float]] = [(lm.x * ff.w, lm.y * ff.h) for lm in ff.landmarks]
        feats = det.process(ff.t, pts)
        if not feats:
            return
        packed = OrderedDict(sorted(feats.items()))
        packed["t"] = ff.t
        rows_sink.append(packed)

    return on_frame

#################################################################################################################


def make_overlay_callback(args: argparse.Namespace, sensor: FaceSensor, det: Optional[MicroExpDetector] = None):
    """
    Cria o callback responsável por desenhar o overlay visual em cada frame.

    O callback resultante delega o desenho para `draw_micro_overlay`, que é
    quem de fato conhece o layout visual (mesh, textos, eventos, IDs, etc.).

    Parâmetros
    ----------
    args : argparse.Namespace
        Namespace com flags de overlay:
        - show_all : desenhar todos os pontos/landmarks
        - show_ids : desenhar o índice numérico em cada ponto
        - events   : exibir informações sobre eventos detectados
    sensor : FaceSensor
        Instância do sensor de face. (Atualmente não é usada no callback,
        mas é recebida para manter a assinatura flexível para futuras extensões.)
    det : MicroExpDetector | None
        Detector de microexpressões. Também não é usado diretamente aqui,
        mas pode ser útil para overlays mais ricos (p.ex. desenhar estado interno).

    Retorna
    -------
    Callable[[Any, FaceFrame], None]
        Função `overlay_cb(frame, ff: FaceFrame) -> None` para ser registrada via `sensor.set_overlay`.
    """
    def overlay_cb(frame, ff: FaceFrame) -> None:
        draw_micro_overlay(
            frame,
            ff,
            show_all=args.show_all,
            show_ids=args.show_ids,
            show_events=args.events,   # só mostra se --events
        )

    return overlay_cb

#################################################################################################################


def save_outputs(rows: List[OrderedDict], args: argparse.Namespace) -> None:
    """
    Salva as features acumuladas (rows) em disco, nos formatos solicitados.

    Parâmetros
    ----------
    rows : list[OrderedDict]
        Lista de linhas de features produzidas pelo callback de processamento.
        Cada entrada deve ser um OrderedDict com chaves numéricas ou string
        e pelo menos o campo 't' (timestamp).
    args : argparse.Namespace
        Namespace com caminhos de saída:
        - csv     : caminho para salvar como CSV (se string não vazia)
        - parquet : caminho para salvar como Parquet (se string não vazia)

    Efeitos colaterais
    ------------------
    - Cria arquivos CSV e/ou Parquet se paths forem fornecidos.
    - Imprime no stdout um resumo dos arquivos gerados.
    """
    if not rows:
        return
    df = pd.DataFrame(rows).fillna(0.0)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"[run_micro] CSV salvo em: {args.csv}  ({len(df)} linhas)")
    if args.parquet:
        df.to_parquet(args.parquet, index=False)
        print(f"[run_micro] Parquet salvo em: {args.parquet}  ({len(df)} linhas)")

#################################################################################################################


def main(argv=None) -> None:
    """
    Função principal do módulo de coleta de microexpressões.

    Fluxo geral:
    1. Faz o parse dos argumentos de linha de comando.
    2. Constrói o MicroExpDetector e o FaceSensor.
    3. Cria e registra o callback de features (on_frame).
    4. Cria e registra o callback de overlay (set_overlay).
    5. Inicia o loop de captura com `sensor.start()`.
    6. Ao término (normal ou por exceção), persiste as features geradas.

    Parâmetros
    ----------
    argv : list[str] | None
        Lista de argumentos a serem parseados. Se None, usa sys.argv.

    Efeitos colaterais
    ------------------
    - Abre a câmera e inicia o loop de captura.
    - Pode exibir preview com overlay em uma janela de vídeo, se `--preview`.
    - Gera arquivos de saída (CSV/Parquet) se caminhos forem fornecidos.
    """
    args   = parse_args(argv)
    det    = build_detector(args)
    sensor = build_sensor(args)

    rows: List[OrderedDict] = []
    sensor.on_frame(make_feature_callback(det, rows))
    sensor.set_overlay(make_overlay_callback(args, sensor, det))

    try:
        sensor.start()
    finally:
        save_outputs(rows, args)

#################################################################################################################

if __name__ == "__main__":
    main()
