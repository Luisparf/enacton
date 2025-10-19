import time
from typing import Callable, Optional, List

import cv2
import mediapipe as mp

from .contracts import Landmark, FaceFrame, now_ms


class FaceSensor:
    """
    Captura frames de vídeo da webcam e extrai landmarks faciais usando MediaPipe FaceMesh.

    Responsabilidades (Embodied):
        - Ler frames da câmera via OpenCV.
        - Processar cada frame com MediaPipe FaceMesh (com `refine_landmarks=True` para incluir íris).
        - Converter os pontos em objetos `Landmark` definidos em contracts.py.
        - Montar um `FaceFrame` com todos os landmarks + íris e entregar via callback.
        - Opcionalmente exibir preview com overlay limpo.

    Atributos:
        cap (cv2.VideoCapture | None): handler da câmera.
        cb (Optional[Callable[[FaceFrame], None]]): callback chamado a cada frame processado.
        enable_preview (bool): se True, exibe janela de preview do rosto.
        cam_index (int): índice da câmera a usar (default: 0).
        width (int): largura da captura pedida ao driver.
        height (int): altura da captura pedida ao driver.
        facemesh (mp.solutions.face_mesh.FaceMesh): modelo de rastreamento facial MediaPipe.

    Uso:
        sensor = FaceSensor(enable_preview=True)
        sensor.on_frame(lambda ff: print(ff))
        sensor.start()
    """

    def __init__(
        self,
        cam_index:      int   = 0,
        width:          int   = 640,
        height:         int   = 480,
        enable_preview: bool  = True,
        min_det_conf:   float = 0.7,
        min_track_conf: float = 0.7,
        overlay_style:  str   = "hud",            # "contours" | "oval" | "boxes" | "hud"
        line_thickness: int   = 2,               # espessura dos traços
        color_face:     tuple[int,int,int] = (255, 255, 255),   # contorno rosto (B,G,R)
        color_boxes:    tuple[int,int,int] = (200, 255, 255),  # caixas olhos
        color_lips:     tuple[int,int,int] = (255, 255, 255),   # lábios
        color_iris:     tuple[int,int,int] = (0, 255, 255),     # íris/crosshair
        overlay_alpha:  float = 1.0,            # 0..1 (1 = desenha direto; <1 = blend suave)
    ):
        """
        Inicializa o FaceSensor.

        Args:
            cam_index: índice da câmera no OpenCV (default: 0).
            width: largura desejada da captura (default: 640).
            height: altura desejada da captura (default: 480).
            enable_preview: se True, exibe janela de preview com landmarks.
            min_det_conf: confiança mínima de detecção.
            min_track_conf: confiança mínima de tracking.
        """
        self.cap = None
        self.cb: Optional[Callable[[FaceFrame], None]] = None
        self.enable_preview = enable_preview
        self.cam_index, self.width, self.height = cam_index, width, height
        self.overlay_style  = overlay_style
        self.line_thickness = int(line_thickness)
        self.color_face     = tuple(color_face)
        self.color_boxes    = tuple(color_boxes)
        self.color_lips     = tuple(color_lips)
        self.color_iris     = tuple(color_iris)
        self.overlay_alpha  = float(overlay_alpha)

        #  Configuração estável (a que funcionou): refine_landmarks=True e max_num_faces=1
        self.facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,          # adiciona os 5 pontos da íris por olho
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

    # ---------------- API pública ----------------

    def on_frame(self, cb: Callable[[FaceFrame], None]) -> None:
        """Registra o callback a ser chamado a cada `FaceFrame`."""
        self.cb = cb

    def _enable_cv_optimizations(self, num_threads: int = 4) -> None:
        """Liga otimizações do OpenCV e ajusta nº de threads."""
        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(num_threads)
        except Exception:
            pass


    def start(self) -> None:
        """
        Inicia o loop de captura e processamento facial.

        - Lê frames da câmera.
        - Extrai landmarks faciais (468 pontos + íris).
        - Constrói um `FaceFrame` e envia ao callback.
        - Se `enable_preview=True`, mostra janela com preview (tecla 'q' para sair).

        Levanta:
            RuntimeError: se a câmera não puder ser aberta.
        """
        self._enable_cv_optimizations(8)
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera (index={self.cam_index}).")

        # Tenta ajustar resolução (nem todos os drivers respeitam)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width ) 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        no_face_counter = 0

        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[FaceSensor] Falha ao ler frame da câmera.")
                break

            # MediaPipe espera RGB; não espelhar para evitar confusão no gaze
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = self.facemesh.process(rgb)
            lms: List[Landmark] = []
            iris_l, iris_r = [], []

            if res.multi_face_landmarks:
                no_face_counter = 0
                face = res.multi_face_landmarks[0]

                # Copia landmarks para nosso contrato
                for lm in face.landmark:
                    lms.append(Landmark(lm.x, lm.y, lm.z, 1.0))

                # Índices da íris quando refine_landmarks=True:
                # direita: 468..472 | esquerda: 473..477
                if len(lms) >= 478:
                    iris_r = [lms[i] for i in range(468, 473)]
                    iris_l = [lms[i] for i in range(473, 478)]

                # --- DESENHO LIMPO (contornos + íris) ---
                if self.enable_preview:
                   self._draw_overlay(frame, face, iris_l, iris_r)  # usa configs


            else:
                no_face_counter += 1
                if no_face_counter % 30 == 0:
                    print("[FaceSensor] Nenhum rosto detectado (verifique luz/ângulo/óculos).")

            # Emite ao pipeline
            ff = FaceFrame(t=now_ms(), landmarks=lms, iris_left=iris_l, iris_right=iris_r)
            if self.cb:
                self.cb(ff)

            if self.enable_preview:
                cv2.imshow("Face Preview (q para sair)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self._cleanup()

    # ---------------- Internos ----------------

    def _draw_overlay(self, frame, face_landmarks, iris_l, iris_r) -> None:
        """
        Overlay apresentável com múltiplos estilos, parametrizado por:
        - self.overlay_style
        - self.line_thickness, self.color_face, self.color_boxes, self.color_lips, self.color_iris
        - self.overlay_alpha (0..1)
        """
        import numpy as np
        import cv2
        mpfm = mp.solutions.face_mesh
        draw = mp.solutions.drawing_utils

        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        lt = max(1, self.line_thickness)

        # desenhar em overlay para permitir alpha blend suave
        base = frame
        overlay = frame.copy()

        def poly(idxs):
            return np.array([pts[i] for i in idxs], dtype=np.int32).reshape(-1, 1, 2)

        def rect(idxs):
            xs = [pts[i][0] for i in idxs]; ys = [pts[i][1] for i in idxs]
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
            OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            cv2.polylines(overlay, [poly(OVAL)], True, self.color_face, lt, lineType=cv2.LINE_AA)

        elif style == "boxes":
            LEFT_EYE  = [33,160,158,133,153,144]
            RIGHT_EYE = [263,387,385,362,380,373]
            MOUTH     = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,291]
            for idxs in (LEFT_EYE, RIGHT_EYE):
                x1,y1,x2,y2 = rect(idxs)
                cv2.rectangle(overlay, (x1,y1), (x2,y2), self.color_boxes, lt, lineType=cv2.LINE_AA)
            cv2.polylines(overlay, [poly(MOUTH)], True, self.color_lips, lt, lineType=cv2.LINE_AA)
        elif style == "boxes_brows":
            # --- Retângulos na face/olhos/boca + linhas retas nas sobrancelhas ---
            # FaceMesh subsets
            LEFT_EYE       = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE      = [263, 387, 385, 362, 380, 373]
            LIPS_OUTER     = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317]
            # Sobrancelhas (conectaremos com segmentos retos)
            LEFT_BROW      = [70, 63, 105, 66, 107]           # aprox. supercílios esquerdo
            RIGHT_BROW     = [336, 296, 334, 293, 300]        # aprox. supercílios direito
            # Face "oval" para retângulo de face
            OVAL           = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                            152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

            # helpers locais herdando configs do sensor
            def rect_from_idxs(idxs):
                xs         = [pts[i][0] for i in idxs]
                ys         = [pts[i][1] for i in idxs]
                return min(xs), min(ys), max(xs), max(ys)

            # 1) retângulo da face
            x1, y1, x2, y2  = rect_from_idxs(OVAL)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_face, lt, lineType=cv2.LINE_AA)

            # 2) retângulos dos olhos
            for idxs in (LEFT_EYE, RIGHT_EYE):
                x1, y1, x2, y2  = rect_from_idxs(idxs)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_boxes, lt, lineType=cv2.LINE_AA)

            # 3) retângulo da boca (a partir do contorno externo)
            x1, y1, x2, y2  = rect_from_idxs(LIPS_OUTER)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_lips, lt, lineType=cv2.LINE_AA)

            # 4) linhas retas conectando pontos das sobrancelhas
            def draw_brow_lines(idxs, color):
                for i in range(len(idxs) - 1):
                    p1     = pts[idxs[i]]
                    p2     = pts[idxs[i + 1]]
                    cv2.line(overlay, p1, p2, color, lt, lineType=cv2.LINE_AA)

            draw_brow_lines(LEFT_BROW,  self.color_boxes)
            draw_brow_lines(RIGHT_BROW, self.color_boxes)

        else:  # "hud" default: oval + boxes + lábios
            OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            LEFT_EYE  = [33,160,158,133,153,144]
            RIGHT_EYE = [263,387,385,362,380,373]
            LIPS = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,291]

            cv2.polylines(overlay, [poly(OVAL)], True, self.color_face, lt, lineType=cv2.LINE_AA)
            for idxs in (LEFT_EYE, RIGHT_EYE):
                x1,y1,x2,y2 = rect(idxs)
                cv2.rectangle(overlay, (x1,y1), (x2,y2), self.color_boxes, lt, lineType=cv2.LINE_AA)
            cv2.polylines(overlay, [poly(LIPS)], True, self.color_lips, lt, lineType=cv2.LINE_AA)

        # íris em qualquer estilo
        for group in (iris_l or []), (iris_r or []):
            for lm in group:
                cx, cy = int(lm.x * w), int(lm.y * h)
                iris_crosshair(overlay, cx, cy, r=5)

        # aplica alpha (blend) se < 1.0
        a = float(self.overlay_alpha)
        if a < 1.0:
            cv2.addWeighted(overlay, a, base, 1.0 - a, 0, base)
        else:
            base[:] = overlay


    def _cleanup(self) -> None:
        """Libera recursos (câmera, janelas OpenCV, modelo MediaPipe)."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.facemesh.close()



   
