import time
from typing import Callable, Optional, List, Tuple

import cv2
import mediapipe as mp

from .contracts   import Landmark, FaceFrame, now_ms
from .emotion_net import EmotionNet  

class FaceSensor:
    """
    Captura frames da webcam, extrai landmarks com MediaPipe FaceMesh,
    renderiza um overlay limpo (vários estilos) e, opcionalmente, prediz
    emoção facial em tempo real usando EmotionNet (ResNet local).

    Responsabilidades (Embodied):
      - Ler frames da câmera via OpenCV.
      - Processar cada frame com MediaPipe FaceMesh (refine_landmarks=True).
      - Converter os landmarks em `Landmark` (contracts.py) e montar `FaceFrame`.
      - Renderizar overlay suave/anti-aliased.
      - (Opcional) Classificar emoção com modelo ResNet local (EmotionNet).

    Atributos principais:
      cap              : handler de câmera (cv2.VideoCapture).
      facemesh         : MediaPipe FaceMesh (com íris).
      emotion_net      : classificador de emoção (ResNet local).
      overlay_*        : parâmetros visuais (cores, espessura, alpha).
      debug_indices    : exibe os índices dos landmarks escolhidos.
    """

    def __init__(
        self,
        cam_index:       int                 = 0,
        width:           int                 = 640,
        height:          int                 = 480,
        enable_preview:  bool                = True,
        min_det_conf:    float               = 0.70,
        min_track_conf:  float               = 0.70,
        overlay_style:   str                 = "boxes_brows",   # "contours" | "oval" | "boxes" | "hud" | "boxes_brows"
        line_thickness:  int                 = 2,
        color_face:      tuple[int,int,int]  = (255, 255, 255),
        color_boxes:     tuple[int,int,int]  = (200, 255, 255),
        color_lips:      tuple[int,int,int]  = (255, 255, 255),
        color_iris:      tuple[int,int,int]  = (0, 255, 255),
        overlay_alpha:   float               = 1.0,             # 1.0 = desenha direto; <1.0 = blend com base
        debug_indices:   bool                = True,            # desenha números dos landmarks selecionados
        enable_emotion:  bool                = True,            # controle explícito da emoção (ResNet local)
    ):
        """
        Inicializa o FaceSensor.

        Args:
            cam_index       : índice da câmera (default: 0).
            width / height  : resolução sugerida (o driver pode ajustar).
            enable_preview  : se True, abre janela com overlay.
            min_det_conf    : confiança mínima de detecção (MediaPipe).
            min_track_conf  : confiança mínima de tracking (MediaPipe).
            overlay_style   : estilo do overlay (ver opções acima).
            line_thickness  : espessura das linhas desenhadas.
            color_*         : cores (B,G,R) para rosto/caixas/lábios/íris.
            overlay_alpha   : alpha do overlay (0..1).
            debug_indices   : desenhar índices dos landmarks (debug visual).
            enable_emotion  : ativa a inferência de emoção via EmotionNet (ResNet local).
        """
        self.cap                   = None
        self.cb: Optional[Callable[[FaceFrame], None]] = None

        self.enable_preview        = bool(enable_preview)
        self.cam_index             = int(cam_index)
        self.width                 = int(width)
        self.height                = int(height)

        # Visual / overlay
        self.overlay_style         = str(overlay_style)
        self.line_thickness        = int(line_thickness)
        self.color_face            = tuple(color_face)
        self.color_boxes           = tuple(color_boxes)
        self.color_lips            = tuple(color_lips)
        self.color_iris            = tuple(color_iris)
        self.overlay_alpha         = float(overlay_alpha)

        # Debug de índices
        self.debug_indices         = bool(debug_indices)
        self.debug_color           = (0, 255, 255)
        self.debug_font_scale      = 0.35
        self.debug_radius          = 2

        # Emoção (ResNet local)
        self.enable_emotion        = bool(enable_emotion)
        self.emotion_net: Optional[EmotionNet] = None
        self.last_emotion: Tuple[str, float] = ("neutral", 0.0)
        if self.enable_emotion:
            # Presume que EmotionNet lida com carregamento de pesos localmente
            # (sem downloads) e expõe .predict(frame_bgr, bbox_xyxy) -> (label, conf, logits/extra)
            self.emotion_net       = EmotionNet(
                                        weights_path="weights/emotion_resnet18_fer2013.pth",  # use o seu .pth real!
                                        smooth_window=1,          # sem suavização p/ medir conf real
                                        do_equalize=False,        # desliga equalização por enquanto
                                        min_face_size=48,         # baixa um pouco
                                        min_conf=0.05,            # baixa threshold pra não virar unknown à toa
                                        device="auto",
                                        verbose=True              # loga motivos
                                    )

        # MediaPipe FaceMesh (inclui íris com refine_landmarks=True)
        self.facemesh              = mp.solutions.face_mesh.FaceMesh(
            static_image_mode         = False,
            max_num_faces             = 1,
            refine_landmarks          = True,                  # adiciona 5 pts de íris por olho
            min_detection_confidence  = float(min_det_conf),
            min_tracking_confidence   = float(min_track_conf),
        )

    # -------------------------------------------------------------------------
    # API pública
    # -------------------------------------------------------------------------

    def on_frame(self, cb: Callable[[FaceFrame], None]) -> None:
        """Registra callback a ser chamado a cada `FaceFrame` produzido."""
        self.cb                   = cb

    def _enable_cv_optimizations(self, num_threads: int = 4) -> None:
        """Liga otimizações do OpenCV e ajusta nº de threads (quando suportado)."""
        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(int(num_threads))
        except Exception:
            pass

    def start(self) -> None:
        """
        Inicia o loop de captura e processamento facial.

        Pipeline:
          - Lê frame (BGR) da câmera
          - Converte para RGB
          - Processa FaceMesh → landmarks (468 + íris)
          - Monta `FaceFrame` e envia via callback
          - Renderiza overlay + (opcional) emoção (ResNet local)
          - Mostra janela (se `enable_preview=True`, tecla 'q' encerra)

        Raises:
            RuntimeError: se a câmera não puder ser aberta.
        """
        self._enable_cv_optimizations(8)

        self.cap                 = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FPS, 30)                      # alvo de FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera (index={self.cam_index}).")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        no_face_counter          = 0
        t0                       = time.time()
        fps_est                  = 0.0

        while True:
            ok, frame            = self.cap.read()
            if not ok:
                print("[FaceSensor] Falha ao ler frame da câmera.")
                break

            rgb                  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res                  = self.facemesh.process(rgb)

            lms: List[Landmark]  = []
            iris_l, iris_r       = [], []

            if res.multi_face_landmarks:
                no_face_counter  = 0
                face             = res.multi_face_landmarks[0]

                for lm in face.landmark:
                    lms.append(Landmark(lm.x, lm.y, lm.z, 1.0))

                if len(lms) >= 478:
                    iris_r       = [lms[i] for i in range(468, 473)]
                    iris_l       = [lms[i] for i in range(473, 478)]

                if self.enable_preview:
                    self._draw_overlay(frame, face, iris_l, iris_r)

            else:
                no_face_counter += 1
                if no_face_counter % 30 == 0:
                    print("[FaceSensor] Nenhum rosto detectado (verifique luz/ângulo/óculos).")

            ff                    = FaceFrame(t=now_ms(), landmarks=lms, iris_left=iris_l, iris_right=iris_r)
            if self.cb:
                self.cb(ff)

            if self.enable_preview:
                cv2.imshow("Face Preview (q para sair)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            dt                    = max(time.time() - t0, 1e-6)
            fps_inst              = 1.0 / dt
            fps_est               = (fps_est * 0.9) + (fps_inst * 0.1)
            t0                    = time.time()
            cv2.putText(frame, f"{fps_est:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self._cleanup()

    # -------------------------------------------------------------------------
    # Internos
    # -------------------------------------------------------------------------

    def _draw_overlay(self, frame, face_landmarks, iris_l, iris_r) -> None:
        """
        Desenha overlay apresentável com múltiplos estilos e, em seguida,
        (se habilitado) calcula e desenha a **emoção** usando EmotionNet (ResNet).

        Notas:
          - os índices (quando habilitados) são desenhados **antes** do alpha-blend.
        """
        import numpy as np
        mpfm                  = mp.solutions.face_mesh
        draw                  = mp.solutions.drawing_utils

        h, w                  = frame.shape[:2]
        pts                   = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        lt                    = max(1, self.line_thickness)

        base                  = frame
        overlay               = frame.copy()

        def poly(idxs):
            return np.array([pts[i] for i in idxs], dtype=np.int32).reshape(-1, 1, 2)

        def rect_from_pts(idxs):
            xs               = [pts[i][0] for i in idxs]
            ys               = [pts[i][1] for i in idxs]
            return min(xs), min(ys), max(xs), max(ys)

        def iris_crosshair(img, cx, cy, r=6):
            cv2.circle(img, (cx, cy), 2, self.color_iris, -1, lineType=cv2.LINE_AA)
            cv2.line(img, (cx - r, cy), (cx + r, cy), self.color_iris, 1, lineType=cv2.LINE_AA)
            cv2.line(img, (cx, cy - r), (cx, cy + r), self.color_iris, 1, lineType=cv2.LINE_AA)

        style                 = self.overlay_style

        if style == "contours":
            draw.draw_landmarks(
                image                       = overlay,
                landmark_list               = face_landmarks,
                connections                 = mpfm.FACEMESH_CONTOURS,
                landmark_drawing_spec       = None,
                connection_drawing_spec     = draw.DrawingSpec(color=self.color_face, thickness=lt, circle_radius=0),
            )

        elif style == "oval":
            OVAL               = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                                  152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            cv2.polylines(overlay, [poly(OVAL)], True, self.color_face, lt, lineType=cv2.LINE_AA)

        elif style == "boxes":
            LEFT_EYE          = [33,160,158,133,153,144]
            RIGHT_EYE         = [263,387,385,362,380,373]
            MOUTH             = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,291]
            for idxs in (LEFT_EYE, RIGHT_EYE):
                x1, y1, x2, y2 = rect_from_pts(idxs)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_boxes, lt, lineType=cv2.LINE_AA)
            cv2.polylines(overlay, [poly(MOUTH)], True, self.color_lips, lt, lineType=cv2.LINE_AA)

            if self.debug_indices:
                self._draw_indices(overlay, pts, LEFT_EYE)

        elif style == "boxes_brows":
            LEFT_EYE          = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE         = [263, 387, 385, 362, 380, 373]
            LIPS_OUTER        = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317]
            LEFT_BROW         = [70, 63, 105, 66, 107]
            RIGHT_BROW        = [336, 296, 334, 293, 300]
            OVAL              = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                                 152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

            def draw_brow_lines(idxs, color):
                for i in range(len(idxs) - 1):
                    p1       = pts[idxs[i]]
                    p2       = pts[idxs[i + 1]]
                    cv2.line(overlay, p1, p2, color, lt, lineType=cv2.LINE_AA)

            x1, y1, x2, y2     = rect_from_pts(OVAL)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_face, lt, lineType=cv2.LINE_AA)
            for idxs in (LEFT_EYE, RIGHT_EYE):
                ex1, ey1, ex2, ey2 = rect_from_pts(idxs)
                cv2.rectangle(overlay, (ex1, ey1), (ex2, ey2), self.color_boxes, lt, lineType=cv2.LINE_AA)
            bx1, by1, bx2, by2 = rect_from_pts(LIPS_OUTER)
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), self.color_lips, lt, lineType=cv2.LINE_AA)

            draw_brow_lines(LEFT_BROW,  self.color_boxes)
            draw_brow_lines(RIGHT_BROW, self.color_boxes)

        else:  # "hud"
            OVAL               = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                                  152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            LEFT_EYE           = [33,160,158,133,153,144]
            RIGHT_EYE          = [263,387,385,362,380,373]
            LIPS               = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,291]

            cv2.polylines(overlay, [poly(OVAL)], True, self.color_face, lt, lineType=cv2.LINE_AA)
            for idxs in (LEFT_EYE, RIGHT_EYE):
                x1, y1, x2, y2 = rect_from_pts(idxs)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_boxes, lt, lineType=cv2.LINE_AA)
            cv2.polylines(overlay, [poly(LIPS)], True, self.color_lips, lt, lineType=cv2.LINE_AA)

        # Íris (crosshair)
        for group in (iris_l or []), (iris_r or []):
            for lm in group:
                cx, cy          = int(lm.x * w), int(lm.y * h)
                iris_crosshair(overlay, cx, cy, r=5)

        # Emoção (ResNet local) — usa bbox do contorno facial (OVAL)
        if self.enable_emotion and self.emotion_net is not None:
            OVAL_BBOX        = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                                152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            try:
                fx1, fy1, fx2, fy2 = rect_from_pts(OVAL_BBOX)
                label, conf, _     = self.emotion_net.predict(frame, (fx1, fy1, fx2, fy2))
                self.last_emotion  = (label, conf)
                text               = f"{label.upper()} ({conf*100:.1f}%)"
                cv2.putText(overlay, text, (fx1, max(20, fy1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                # silencioso por robustez
                pass

        # Índices de debug
        if self.debug_indices:
            LEFT_EYE_DI        = [33,160,158,133,153,144]
            RIGHT_EYE_DI       = [263,387,385,362,380,373]
            KEY_MOUTH          = [61, 291, 13, 14, 78, 308, 82, 312]
            KEY_BROW_LEFT      = [70, 63, 105, 66]
            KEY_BROW_RIGHT     = [336, 296, 334, 293]
            self._draw_indices(overlay, pts, LEFT_EYE_DI)
            self._draw_indices(overlay, pts, RIGHT_EYE_DI)
            self._draw_indices(overlay, pts, KEY_MOUTH)
            self._draw_indices(overlay, pts, KEY_BROW_LEFT)
            self._draw_indices(overlay, pts, KEY_BROW_RIGHT)

        a                     = float(self.overlay_alpha)
        if a < 1.0:
            cv2.addWeighted(overlay, a, base, 1.0 - a, 0, base)
        else:
            base[:]           = overlay

    def _cleanup(self) -> None:
        """Libera câmera, fecha janelas e encerra o modelo do MediaPipe."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.facemesh.close()

    def _draw_indices(self, overlay, pts, idxs) -> None:
        """Desenha marcadores e os números dos índices (debug visual)."""
        for i in idxs:
            if i >= len(pts):
                continue
            cx, cy             = pts[i]
            cv2.circle(overlay, (cx, cy), self.debug_radius, self.debug_color, -1, lineType=cv2.LINE_AA)
            cv2.putText(overlay, str(i), (cx + 3, cy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, self.debug_font_scale,
                        self.debug_color, 1, cv2.LINE_AA)
