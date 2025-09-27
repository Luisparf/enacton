import cv2, mediapipe as mp
from typing import Callable, Optional, List
from .contracts import Landmark, FaceFrame, now_ms


class FaceSensor:
    """
    Captura frames de vídeo da webcam e extrai landmarks faciais usando MediaPipe FaceMesh.

    Responsabilidades (Embodied):
        - Ler frames da câmera via OpenCV.
        - Processar cada frame com MediaPipe FaceMesh (com `refine_landmarks=True` para incluir íris).
        - Converter os pontos em objetos `Landmark` definidos em contracts.py.
        - Montar um `FaceFrame` com todos os landmarks + íris e entregar via callback.
        - Opcionalmente exibir preview com OpenCV.

    Atributos:
        cap (cv2.VideoCapture | None): handler da câmera.
        cb (Optional[Callable[[FaceFrame], None]]): callback chamado a cada frame processado.
        enable_preview (bool): se True, exibe janela de preview do rosto.
        cam_index (int): índice da câmera a usar (default: 0).
        width (int): largura da captura pedida ao driver.
        height (int): altura da captura pedida ao driver.
        facemesh (mp.solutions.face_mesh.FaceMesh): modelo de rastreamento facial MediaPipe.
        draw (mp.solutions.drawing_utils): utilitário para desenhar landmarks no preview.

    Uso:
        sensor = FaceSensor(enable_preview=True)
        sensor.on_frame(lambda ff: print(ff))
        sensor.start()
    """

    def __init__(self, cam_index=0, width=640, height=480, enable_preview=False):
        """
        Inicializa o FaceSensor.

        Args:
            cam_index (int): índice da câmera no OpenCV (default: 0).
            width (int): largura desejada da captura (default: 640).
            height (int): altura desejada da captura (default: 480).
            enable_preview (bool): se True, exibe janela de preview com landmarks.
        """
        self.cap = None
        self.cb: Optional[Callable[[FaceFrame], None]] = None
        self.enable_preview = enable_preview
        self.cam_index, self.width, self.height = cam_index, width, height
        self.facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)  # inclui íris
        self.draw = mp.solutions.drawing_utils

    def on_frame(self, cb: Callable[[FaceFrame], None]):
        """
        Registra o callback a ser chamado a cada `FaceFrame`.

        Args:
            cb (Callable[[FaceFrame], None]): função a ser chamada com o frame processado.
        """
        self.cb = cb

    def start(self):
        """
        Inicia o loop de captura e processamento facial.

        - Lê frames da câmera.
        - Extrai landmarks faciais (468 pontos + íris).
        - Constrói um `FaceFrame` e envia ao callback.
        - Se `enable_preview=True`, mostra janela com preview (tecla 'q' para sair).
        """
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        while True:
            ok, frame = self.cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.facemesh.process(rgb)
            lms: List[Landmark] = []
            iris_l, iris_r = [], []

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                for i, lm in enumerate(face.landmark):
                    lms.append(Landmark(lm.x, lm.y, lm.z, 1.0))
                # MediaPipe FaceMesh (refine_landmarks=True) adiciona pontos da íris
                iris_l = [lms[i] for i in range(473, 478)] if len(lms) >= 478 else []
                iris_r = [lms[i] for i in range(468, 473)] if len(lms) >= 473 else []

            ff = FaceFrame(t=now_ms(), landmarks=lms, iris_left=iris_l, iris_right=iris_r)
            if self.cb:
                self.cb(ff)

            if self.enable_preview:
                cv2.imshow("Face Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self._cleanup()

    def _cleanup(self):
        """
        Libera recursos (câmera, janelas OpenCV, modelo MediaPipe).
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.facemesh.close()
