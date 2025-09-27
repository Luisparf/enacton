# sensor.py
from __future__ import annotations

import time
from typing import Callable, Optional, List

import cv2
import mediapipe as mp

from .contracts import PoseFrame, Landmark, now_ms


class PoseSensor:
    """
    Captura frames da webcam e extrai landmarks de corpo e mãos via MediaPipe.

    Responsabilidades (Embodied):
        - Ler vídeo da câmera (OpenCV).
        - Rodar MediaPipe Pose + Hands para obter landmarks normalizados.
        - Agregar resultado em um PoseFrame e disparar callback do pipeline.

    Parâmetros:
        cam_index (int): índice da câmera no OpenCV (default: 0).
        width (int): largura do frame solicitado ao driver (default: 640).
        height (int): altura do frame solicitada ao driver (default: 480).
        enable_preview (bool): exibe janela de preview (default: True).
        target_fps (Optional[float]): limita taxa de processamento. Se None, roda no máximo possível.
        pose_complexity (int): 0, 1 ou 2 (quanto maior, mais preciso e pesado).
        min_det_conf (float): confiança mínima de detecção inicial (0..1).
        min_track_conf (float): confiança mínima de tracking (0..1).

    Uso:
        sensor = PoseSensor()
        sensor.on_frame(lambda pf: ... )  # registra callback
        sensor.start()                    # loop bloqueante até stop() ou 'q' no preview

    Notas:
        - Coordenadas de landmarks do MediaPipe são normalizadas (0..1) no espaço da imagem.
        - A ordem dos pontos segue o modelo MediaPipe (Pose: 33; Hands: 21).
        - Este módulo NÃO faz suavização/EMA: isso é função do encoder/engine.
    """

    def __init__(
        self,
        cam_index: int = 0,
        width: int = 640,
        height: int = 480,
        enable_preview: bool = True,
        target_fps: Optional[float] = None,
        pose_complexity: int = 1,
        min_det_conf: float = 0.5,
        min_track_conf: float = 0.5,
    ):
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.enable_preview = enable_preview
        self.target_fps = target_fps

        # callback do pipeline (Runtime)
        self._cb: Optional[Callable[[PoseFrame], None]] = None

        # controle de execução
        self._running: bool = False
        self._last_tick: float = 0.0
        self._period: Optional[float] = (1.0 / target_fps) if target_fps else None

        # OpenCV
        self._cap: Optional[cv2.VideoCapture] = None

        # MediaPipe solutions
        self._mp_pose = mp.solutions.pose.Pose(
            model_complexity=pose_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )
        self._mp_hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )
        self._mp_drawing = mp.solutions.drawing_utils

    # ----------------------------
    # API pública
    # ----------------------------

    def on_frame(self, cb: Callable[[PoseFrame], None]) -> None:
        """Registra o callback a ser chamado a cada PoseFrame produzido."""
        self._cb = cb

    def start(self) -> None:
        """
        Inicia o loop de captura e processamento.

        Bloqueia a thread atual. Para encerrar:
            - chame self.stop(); ou
            - pressione 'q' na janela de preview (se enable_preview=True).
        """
        self._open_camera()
        self._running = True

        try:
            while self._running:
                # rate limiting (se configurado)
                if self._period is not None:
                    now = time.time()
                    wait = self._period - (now - self._last_tick)
                    if wait > 0:
                        time.sleep(wait)
                    self._last_tick = time.time()

                ok, frame = self._cap.read()
                if not ok:
                    continue

                # MediaPipe espera RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # --------- Corpo ---------
                pose_res = self._mp_pose.process(rgb)
                body: List[Landmark] = []
                if pose_res.pose_landmarks:
                    for lm in pose_res.pose_landmarks.landmark:
                        body.append(Landmark(lm.x, lm.y, lm.z, lm.visibility))

                # --------- Mãos ----------
                hands_res = self._mp_hands.process(rgb)
                left, right = [], []
                if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
                    for lmks, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
                        pts = [Landmark(l.x, l.y, l.z, 1.0) for l in lmks.landmark]
                        label = handedness.classification[0].label.lower()
                        # MediaPipe já tenta rotular left/right em espaço da pessoa (não do espelho):
                        if label == "left":
                            left = pts
                        else:
                            right = pts

                # monta frame do contrato
                pf = PoseFrame(
                    t=now_ms(),
                    left_hand=left,
                    right_hand=right,
                    body=body,
                )

                # emite ao pipeline
                if self._cb:
                    self._cb(pf)

                # preview opcional
                if self.enable_preview:
                    self._draw_preview(frame, pose_res, hands_res)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop()
        finally:
            self._cleanup()

    def stop(self) -> None:
        """Solicita parada do loop de captura."""
        self._running = False

    # ----------------------------
    # Internos
    # ----------------------------

    def _open_camera(self) -> None:
        self._cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)  # CAP_DSHOW evita delays no Windows
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera index={self.cam_index}")

        # Tenta ajustar resolução (nem todos os drivers respeitam)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _draw_preview(self, bgr_frame, pose_res, hands_res) -> None:
        """Desenha landmarks sobre o frame BGR e exibe a janela de preview."""
        # desenha pose
        if pose_res.pose_landmarks:
            self._mp_drawing.draw_landmarks(
                bgr_frame,
                pose_res.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
            )
        # desenha mãos
        if hands_res.multi_hand_landmarks:
            for lmks in hands_res.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    bgr_frame,
                    lmks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                )

        cv2.imshow("PoseSensor Preview (q para sair)", bgr_frame)

    def _cleanup(self) -> None:
        """Encerra recursos (janela OpenCV, camera, modelos MediaPipe)."""
        try:
            if self._cap:
                self._cap.release()
            if self.enable_preview:
                cv2.destroyAllWindows()
        finally:
            # fecha MediaPipe explicitamente
            self._mp_pose.close()
            self._mp_hands.close()
