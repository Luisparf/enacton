# mpfs/face_sensor.py
"""
Módulo responsável por captura de vídeo, extração de landmarks faciais
via MediaPipe FaceMesh e despacho de frames/estruturas FaceFrame para
callbacks externos (features, overlay, etc.).

Fluxo típico de uso:
    sensor = FaceSensor(enable_preview=True)
    sensor.set_overlay(lambda frame, ff: draw_micro_overlay(frame, ff))
    sensor.on_frame(lambda ff: process(ff))
    sensor.start()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple

import time
import cv2
import mediapipe as mp

from .contracts import Landmark, FaceFrame, now_ms
from .overlay import draw_micro_overlay, draw_mesh_mediapipe

try:
    # Opcional: só será usado se enable_emotion=True
    from .emotion_net import EmotionNet
    _HAS_EMOTION = True
except Exception:
    _HAS_EMOTION = False


class FaceSensor:
    """
    Sensor de face baseado em webcam + MediaPipe FaceMesh.

    Responsabilidades principais:
    - Abrir a câmera e capturar frames BGR via OpenCV.
    - Rodar o MediaPipe FaceMesh para extrair landmarks (468 pontos + íris).
    - Montar um `FaceFrame` com largura/altura/timestamp/landmarks/íris.
    - Desenhar um overlay leve (HUD, caixas, contornos, estilo MediaPipe, etc.).
    - Disparar callbacks:
        * `on_frame(cb: FaceFrame -> None)` para processamento numérico.
        * `set_overlay(cb: (frame, FaceFrame) -> None)` para overlays extras.

    Uso típico
    ----------
    sensor = FaceSensor(enable_preview=True)
    sensor.on_frame(lambda ff: process_features(ff))
    sensor.set_overlay(lambda frame, ff: draw_micro_overlay(frame, ff))
    sensor.start()  # loop principal, tecla 'q' fecha a janela
    """

    # ----------------------------------------------------------------------------------

    def __init__(
        self,
        cam_index:       int                 = 0,
        width:           int                 = 640,
        height:          int                 = 480,
        enable_preview:  bool                = True,
        min_det_conf:    float               = 0.70,
        min_track_conf:  float               = 0.70,
        overlay_style:   str                 = "hud",     # "contours" | "oval" | "boxes" | "boxes_brows" | "hud" | "mp"
        line_thickness:  int                 = 1,
        color_face:      tuple[int,int,int]  = (255, 255, 255),
        color_boxes:     tuple[int,int,int]  = (220, 255, 255),
        color_lips:      tuple[int,int,int]  = (255, 255, 255),
        color_iris:      tuple[int,int,int]  = (0, 255, 255),
        overlay_alpha:   float               = 1.0,
        debug_indices:   bool                = False,
        # Emoção opcional (mantida por compatibilidade; default OFF)
        enable_emotion:  bool                = False,
        
    ) -> None:
        """
        Inicializa o FaceSensor com parâmetros de captura, overlay e (opcionalmente) emoção.

        Parâmetros
        ----------
        cam_index : int
            Índice da câmera no OpenCV (0, 1, ...).
        width : int
            Largura desejada do frame capturado.
        height : int
            Altura desejada do frame capturado.
        enable_preview : bool
            Se True, abre uma janela de preview com o overlay desenhado.
        min_det_conf : float
            Confiança mínima de detecção para o FaceMesh.
        min_track_conf : float
            Confiança mínima de tracking para o FaceMesh.
        overlay_style : str
            Estilo visual do overlay "leve" desenhado em `_draw_overlay`.
            Pode ser: "contours", "oval", "boxes", "boxes_brows", "hud", "mp".
        line_thickness : int
            Espessura básica das linhas utilizadas no overlay.
        color_face : tuple[int,int,int]
            Cor BGR usada para contornos do rosto.
        color_boxes : tuple[int,int,int]
            Cor BGR usada para caixas (olhos, etc.).
        color_lips : tuple[int,int,int]
            Cor BGR usada para boca/lábios.
        color_iris : tuple[int,int,int]
            Cor BGR usada para marcação das íris.
        overlay_alpha : float
            Alpha para mesclagem entre overlay e frame base.
            - 1.0: desenha diretamente em `frame`.
            - < 1.0: mistura overlay e frame (addWeighted).
        debug_indices : bool
            Se True, desenha índices de alguns landmarks-chave para debug.
        enable_emotion : bool
            Se True e `EmotionNet` estiver disponível, ativa inferência de emoção
            local (não usada por padrão no fluxo de microexpressões).
        """
        self.fps_est = 0.0

        self.cap: Optional[cv2.VideoCapture] = None
        self.cb: Optional[Callable[[FaceFrame], None]] = None

        self.enable_preview        = bool(enable_preview)
        self.cam_index             = int(cam_index)
        self.width                 = int(width)
        self.height                = int(height)

        # Visual
        self.overlay_style         = str(overlay_style)
        self.line_thickness        = int(line_thickness)
        self.color_face            = tuple(color_face)
        self.color_boxes           = tuple(color_boxes)
        self.color_lips            = tuple(color_lips)
        self.color_iris            = tuple(color_iris)
        self.overlay_alpha         = float(overlay_alpha)

        # Debug
        self.debug_indices         = bool(debug_indices)
        self.debug_color           = (0, 255, 255)
        self.debug_font_scale      = 0.35
        self.debug_radius          = 2

        # Hook de overlay extra (ex.: microexpressões)
        self._extra_overlay_cb: Optional[Callable[[any, FaceFrame], None]] = None

        # Emoção local (opcional)
        self.enable_emotion        = bool(enable_emotion)
        self.emotion_net: Optional["EmotionNet"] = None
        self.last_emotion: Tuple[str, float] = ("neutral", 0.0)
        if self.enable_emotion and _HAS_EMOTION:
            # Se for usar emoção, ajuste o caminho do .pth conforme o seu projeto
            self.emotion_net = EmotionNet(
                weights_path="weights/emotion_resnet18_fer2013.pth",
                smooth_window=1,
                do_equalize=False,
                min_face_size=48,
                min_conf=0.05,
                device="auto",
                verbose=False,
            )

        # MediaPipe FaceMesh
        self.facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,                 # inclui pontos da íris
            min_detection_confidence=float(min_det_conf),
            min_tracking_confidence=float(min_track_conf),
        )

    # ----------------------------------------------------------------------------------
    # API pública
    # ----------------------------------------------------------------------------------

    def set_overlay(self, cb: Callable[[any, FaceFrame], None]) -> None:
        """
        Registra um callback de overlay extra a ser chamado após o overlay padrão.

        O callback recebe:
        - frame: np.ndarray (BGR), modificado in-place.
        - ff   : FaceFrame correspondente ao frame atual.

        Exemplos
        --------
        sensor.set_overlay(
            lambda frame, ff: draw_micro_overlay(frame, ff, show_all=True)
        )
        """
        self._extra_overlay_cb = cb

    def on_frame(self, cb: Callable[[FaceFrame], None]) -> None:
        """
        Registra um callback para processamento de cada FaceFrame produzido.

        O callback recebe apenas um `FaceFrame` (dados, sem overlay) e pode
        ser usado para extração de features, logging, gravação em disco, etc.

        Exemplo
        -------
        sensor.on_frame(lambda ff: print(ff.t, len(ff.landmarks)))
        """
        self.cb = cb

    # ----------------------------------------------------------------------------------

    def start(self) -> None:
        """
        Inicia o loop de captura, detecção facial e overlay.

        Fluxo detalhado por iteração:
        1. Captura um frame BGR da câmera via OpenCV.
        2. Converte para RGB e roda o MediaPipe FaceMesh.
        3. Constrói uma lista de `Landmark` (468 pontos + íris, se presentes).
        4. Se `enable_preview=True`, desenha o overlay "leve" via `_draw_overlay`.
        5. Monta um `FaceFrame` com w/h/t/landmarks/íris.
        6. Chama o callback registrado em `on_frame` (se houver).
        7. Chama o callback registrado em `set_overlay` (se houver), para overlays extras.
        8. Desenha FPS no canto superior esquerdo (se preview).
        9. Exibe a janela "Face Preview" (se preview); tecla 'q' encerra o loop.

        Em caso de falha ao abrir a câmera, lança `RuntimeError`.
        Ao sair, garante liberação da câmera e destruição das janelas via `_cleanup()`.
        """
        self._enable_cv_optimizations(12)

        # Abre câmera e força MJPG (ganho considerável de FPS em muitas webcams)
        self.cap = cv2.VideoCapture(self.cam_index)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        # meta de FPS e resolução
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera (index={self.cam_index}).")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        no_face_counter = 0
        t0 = time.time()
        fps_est = 0.0

        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[FaceSensor] Falha ao ler frame da câmera.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.facemesh.process(rgb)

            lms: List[Landmark] = []
            iris_l: List[Landmark] = []
            iris_r: List[Landmark] = []

            if res.multi_face_landmarks:
                no_face_counter = 0
                face = res.multi_face_landmarks[0]

                for lm in face.landmark:
                    lms.append(Landmark(lm.x, lm.y, lm.z, 1.0))

                if len(lms) >= 478:
                    iris_r = [lms[i] for i in range(468, 473)]
                    iris_l = [lms[i] for i in range(473, 478)]

                if self.enable_preview:
                    self._draw_overlay(frame, face, iris_l, iris_r)
            else:
                no_face_counter += 1
                if no_face_counter % 30 == 0:
                    print("[FaceSensor] Nenhum rosto detectado (verifique luz/ângulo/óculos).")

            h, w = frame.shape[:2]
            ff = FaceFrame(
                w=w, h=h,
                t=now_ms(),
                landmarks=lms,
                iris_left=iris_l,
                iris_right=iris_r
            )

            if self.cb:
                self.cb(ff)

            # Hook para microexpressões (não engole erro)
            if self.enable_preview and self._extra_overlay_cb is not None:
                try:
                    self._extra_overlay_cb(frame, ff)
                except Exception as e:
                    cv2.putText(frame, f"[overlay ERR] {type(e).__name__}: {e}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 180, 255), 2, cv2.LINE_AA)
                    print("[FaceSensor overlay] erro:", repr(e))

            # Preview
            if self.enable_preview:
                # FPS (canto superior esquerdo)
                dt = max(time.time() - t0, 1e-6)
                fps_inst = 1.0 / dt
                fps_est = (fps_est * 0.9) + (fps_inst * 0.1)
                self.fps_est = fps_est
                t0 = time.time()
                cv2.putText(frame, f"{fps_est:.1f} FPS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                            cv2.LINE_AA)

                cv2.imshow("Face Preview (q para sair)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self._cleanup()

    # ----------------------------------------------------------------------------------
    # Internos
    # ----------------------------------------------------------------------------------

    def _enable_cv_optimizations(self, num_threads: int = 4) -> None:
        """
        Liga otimizações do OpenCV e tenta configurar o número de threads.

        Parâmetros
        ----------
        num_threads : int
            Número de threads sugerido para o OpenCV. Se não suportado
            no backend atual, a exceção é ignorada silenciosamente.
        """
        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(int(num_threads))
        except Exception:
            pass

#################################################################################################################

    def _draw_overlay(self, frame, face_landmarks, iris_l, iris_r) -> None:
        """
        Desenha o overlay "leve" padrão diretamente sobre o frame BGR.

        O estilo exato é controlado por `self.overlay_style`:
        - "contours"    : usa FACEMESH_CONTOURS (MediaPipe).
        - "oval"        : polilinha com contorno do rosto.
        - "boxes"       : caixas nos olhos + contorno da boca.
        - "boxes_brows" : caixas + sobrancelhas desenhadas com linhas.
        - "mp"          : delega para `draw_mesh_mediapipe` (mesh completo colorido).
        - "hud" (default): contorno + caixas nos olhos + boca em polilinha.

        Além disso:
        - Desenha crosshair nas íris recebidas (iris_l, iris_r).
        - Opcionalmente desenha índices de debug (`self.debug_indices`).
        - Faz blend com alpha (`overlay_alpha`) para compor com o frame base.

        Parâmetros
        ----------
        frame : np.ndarray
            Frame BGR atual (modificado in-place).
        face_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            Lista de landmarks da face detectada pelo MediaPipe.
        iris_l : list[Landmark]
            Lista de landmarks (normalizados) da íris esquerda (se presentes).
        iris_r : list[Landmark]
            Lista de landmarks (normalizados) da íris direita (se presentes).
        """
        import numpy as np
        mpfm = mp.solutions.face_mesh
        draw = mp.solutions.drawing_utils

        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        lt = max(1, self.line_thickness)

        base = frame
        overlay = frame.copy()

        def poly(idxs):
            return np.array([pts[i] for i in idxs], dtype=np.int32).reshape(-1, 1, 2)

        def rect_from_pts(idxs):
            xs = [pts[i][0] for i in idxs]
            ys = [pts[i][1] for i in idxs]
            return min(xs), min(ys), max(xs), max(ys)

        def iris_crosshair(img, cx, cy, r=6):
            cv2.circle(img, (cx, cy), 2, self.color_iris, -1, lineType=cv2.LINE_AA)
            cv2.line(img, (cx - r, cy), (cx + r, cy), self.color_iris, 1, lineType=cv2.LINE_AA)
            cv2.line(img, (cx, cy - r), (cx, cy + r), self.color_iris, 1, lineType=cv2.LINE_AA)

        style = self.overlay_style

        if style == "contours":
            draw.draw_landmarks(
                image=overlay,
                landmark_list=face_landmarks,
                connections=mpfm.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=draw.DrawingSpec(color=self.color_face, thickness=lt, circle_radius=0),
            )

        elif style == "oval":
            OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                    152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            cv2.polylines(overlay, [poly(OVAL)], True, self.color_face, lt, lineType=cv2.LINE_AA)

        elif style == "boxes":
            LEFT_EYE  = [33,160,158,133,153,144]
            RIGHT_EYE = [263,387,385,362,380,373]
            MOUTH     = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317]
            for idxs in (LEFT_EYE, RIGHT_EYE):
                x1, y1, x2, y2 = rect_from_pts(idxs)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_boxes, lt, lineType=cv2.LINE_AA)
            cv2.polylines(overlay, [poly(MOUTH)], True, self.color_lips, lt, lineType=cv2.LINE_AA)

        elif style == "boxes_brows":
            LEFT_EYE   = [33,160,158,133,153,144]
            RIGHT_EYE  = [263,387,385,362,380,373]
            LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317]
            LEFT_BROW  = [70, 63, 105, 66, 107]
            RIGHT_BROW = [336, 296, 334, 293, 300]
            OVAL       = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                          152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

            def draw_brow(idxs, color):
                for i in range(len(idxs) - 1):
                    cv2.line(overlay, pts[idxs[i]], pts[idxs[i + 1]], color, lt, cv2.LINE_AA)

            x1, y1, x2, y2 = rect_from_pts(OVAL)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_face, lt, cv2.LINE_AA)
            for idxs in (LEFT_EYE, RIGHT_EYE):
                ex1, ey1, ex2, ey2 = rect_from_pts(idxs)
                cv2.rectangle(overlay, (ex1, ey1), (ex2, ey2), self.color_boxes, lt, cv2.LINE_AA)
            bx1, by1, bx2, by2 = rect_from_pts(LIPS_OUTER)
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), self.color_lips, lt, cv2.LINE_AA)

            draw_brow(LEFT_BROW,  self.color_boxes)
            draw_brow(RIGHT_BROW, self.color_boxes)

        elif style == "mp":
            # Estilo MediaPipe customizado (mesh completo com cores por região)
            draw_mesh_mediapipe(overlay, face_landmarks, thickness=1)

        else:  # "hud" (contorno + olhos em caixa + lábios)
            OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                     152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            L_EYE = [33,160,158,133,153,144]
            R_EYE = [263,387,385,362,380,373]
            LIPS  = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317]

            cv2.polylines(overlay, [poly(OVAL)], True, self.color_face, lt, cv2.LINE_AA)
            for idxs in (L_EYE, R_EYE):
                xs = [pts[i][0] for i in idxs]
                ys = [pts[i][1] for i in idxs]
                cv2.rectangle(overlay, (min(xs), min(ys)), (max(xs), max(ys)), self.color_boxes, lt, cv2.LINE_AA)
            cv2.polylines(overlay, [poly(LIPS)], True, self.color_lips, lt, cv2.LINE_AA)

        # Íris
        for group in (iris_l or []), (iris_r or []):
            for lm in group:
                cx, cy = int(lm.x * w), int(lm.y * h)
                iris_crosshair(overlay, cx, cy, r=5)

        # Índices de debug (opcional)
        if self.debug_indices:
            KEY = [33, 133, 263, 362, 61, 291, 13, 14, 70, 63, 336, 296]
            self._draw_indices(overlay, pts, KEY)

        a = float(self.overlay_alpha)
        if a < 1.0:
            cv2.addWeighted(overlay, a, frame, 1.0 - a, 0, frame)
        else:
            frame[:] = overlay

#################################################################################################################

    def _draw_indices(self, overlay, pts, idxs) -> None:
        """
        Desenha círculos e labels numéricos para um conjunto de índices de landmarks.

        Usado para debug visual, por exemplo para marcar alguns pontos importantes
        (olhos, boca, sobrancelhas) sem poluir a tela com todos os 468 índices.

        Parâmetros
        ----------
        overlay : np.ndarray
            Imagem BGR onde os índices serão desenhados (modificada in-place).
        pts : list[tuple[int,int]]
            Lista completa de coordenadas (x, y) dos landmarks.
        idxs : iterable[int]
            Lista/iterável de índices a destacar.
        """
        for i in idxs:
            if i >= len(pts):
                continue
            cx, cy = pts[i]
            cv2.circle(overlay, (cx, cy), self.debug_radius, self.debug_color, -1, cv2.LINE_AA)
            cv2.putText(overlay, str(i), (cx + 3, cy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, self.debug_font_scale,
                        self.debug_color, 1, cv2.LINE_AA)

#################################################################################################################

    def _cleanup(self) -> None:
        """
        Libera recursos associados ao sensor:

        - Fecha a captura de vídeo (`VideoCapture.release()`).
        - Destroi janelas de preview do OpenCV (se `enable_preview=True`).
        - Fecha o objeto FaceMesh do MediaPipe (`self.facemesh.close()`).
        """
        if self.cap:
            self.cap.release()
        if self.enable_preview:
            cv2.destroyAllWindows()
        self.facemesh.close()
