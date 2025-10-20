"""
emotion_net.py
---------------
Detector de emoção facial em tempo real usando **ResNet** local (PyTorch),
pré-treinável/afinável para **FER2013** (7 classes).

Ideia:
    - A detecção de rosto já é feita fora (ex.: MediaPipe FaceMesh).
    - Recebemos um frame BGR (OpenCV) + um bbox (x1,y1,x2,y2) do rosto.
    - Recortamos a face, pré-processamos (opcional equalização + normalização),
      e passamos no modelo ResNet local para obter as probabilidades.

Saída:
    - (label, confidence, probs_dict)
      label       : string com a emoção dominante
      confidence  : probabilidade [0..1] da emoção dominante
      probs_dict  : dicionário com as 7 emoções e suas probabilidades

Emoções (FER2013):
    ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

Uso típico:
    net = EmotionNet(weights_path="weights/emotion_resnet18_fer2013.pth",
                     smooth_window=5, do_equalize=True, device="auto")
    label, conf, probs = net.predict(frame_bgr, bbox=(x1, y1, x2, y2))

Observações:
    - Se não fornecer `weights_path`, o modelo fica com pesos aleatórios (útil só p/ testes de I/O)
      e retornará resultados não confiáveis (poderemos retornar "unknown" se confianca < min_conf).
    - Requer PyTorch/torchvision instalados.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict
from collections import deque
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

"""






 ==================== totalmente instavel essa poha











"""
class EmotionNet:
    """
    Classificador de emoções com ResNet local (PyTorch).

    Parâmetros:
        weights_path  (str|None): caminho para o .pth com pesos da rede (7 classes).
        arch          (str)     : arquitetura base ('resnet18' | 'resnet34' | 'resnet50').
        smooth_window (int)     : tamanho da janela de suavização (>=1). 1 = sem suavização.
        do_equalize   (bool)    : equaliza iluminação (Y/CrCb) no crop.
        bbox_margin   (int)     : margem extra aplicada ao bbox antes do recorte.
        min_face_size (int)     : menor lado aceito no crop; abaixo disso retorna "unknown".
        min_conf      (float)   : confiança mínima para não rotular como "unknown".
        device        (str)     : "auto" | "cpu" | "cuda".
        verbose       (bool)    : logs de debug.

    Atributos:
        labels   (list[str])    : ordem canônica do FER2013.
        smooth_q (deque[Dict])  : janela com distribuições para suavização temporal.
    """

    def __init__(self,
                 weights_path: Optional[str] = None,
                 arch: str = "resnet18",
                 smooth_window: int = 5,
                 do_equalize: bool = True,
                 bbox_margin: int = 12,
                 min_face_size: int = 64,
                 min_conf: float = 0.35,
                 device: str = "auto",
                 verbose: bool = False) -> None:

        if not _HAS_TORCH:
            raise ImportError(
                "[EmotionNet] PyTorch/torchvision não estão instalados. "
                "Instale: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
            )

        self.labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.num_classes = len(self.labels)

        self.smooth_window = int(max(1, smooth_window))
        self.do_equalize = bool(do_equalize)
        self.bbox_margin = int(max(0, bbox_margin))
        self.min_face_size = int(max(16, min_face_size))
        self.min_conf = float(np.clip(min_conf, 0.0, 1.0))
        self.verbose = bool(verbose)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            print(">>> cuda")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            print(">>> cpu")

            self.device = torch.device("cpu")

        # Fila para suavização temporal das probabilidades por classe
        self.smooth_q = deque(maxlen=self.smooth_window)

        # Modelo
        self.model = self._build_model(arch)
        self.model.to(self.device).eval()

        # Carrega pesos (se fornecido)
        self.has_weights = False
        if weights_path:
            try:
                state = torch.load(weights_path, map_location=self.device, weights_only=False)
                # suporta tanto state dict puro quanto checkpoint com 'state_dict'
                if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                    # remover 'module.' caso salvo com DataParallel
                    state = {k.replace("module.", "", 1): v for k, v in state.items()}
                elif isinstance(state, dict) and "state_dict" in state:
                    sd = state["state_dict"]
                    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                    state = sd
                self.model.load_state_dict(state, strict=False)
                self.has_weights = True
                if self.verbose:
                    print(f"[EmotionNet] Pesos carregados de: {weights_path}")
            except Exception as e:
                if self.verbose:
                    print(f"[EmotionNet] Falha ao carregar pesos ({weights_path}): {e}")

        # Transform padrão: RGB 224x224, normalização ImageNet
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        # Warmup leve (opcional)
        # self._warmup()

    # -------------------------------------------------------------------------
    # API pública
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def predict(self,
                frame_bgr: np.ndarray,
                bbox: Tuple[int, int, int, int]) -> Tuple[str, float, Dict[str, float]]:
        """
        Classifica a emoção no rosto recortado pelo bbox.

        Args:
            frame_bgr : frame em BGR (OpenCV), shape [H, W, 3], dtype=uint8.
            bbox      : (x1, y1, x2, y2) em pixels, relativo ao frame.

        Returns:
            label     : string com a emoção dominante (ou "unknown").
            conf      : probabilidade [0..1] da emoção dominante.
            probs     : dicionário {emoção: probabilidade} (vazio se falhar).
        """
        x1, y1, x2, y2 = self._expand_bbox(frame_bgr.shape, bbox, self.bbox_margin)
        face = frame_bgr[y1:y2, x1:x2]

        # Valida crop
        if face.size == 0:
            return "unknown", 0.0, {}

        h, w = face.shape[:2]
        if min(h, w) < self.min_face_size:
            return "unknown", 0.0, {}

        # Pré-processamento
        if self.do_equalize:
            face = self._equalize_lighting(face)

        # BGR -> RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        try:
            tensor = self.tf(face_rgb).unsqueeze(0).to(self.device, non_blocking=True)

            logits = self.model(tensor)  # [1, 7]
            probs_vec = F.softmax(logits, dim=1).float().squeeze(0).cpu().numpy()

            probs = {k: float(v) for k, v in zip(self.labels, probs_vec.tolist())}

            # Suavização temporal
            probs_smooth = self._smooth_probs(probs) if self.smooth_window > 1 else probs

            # Argmax
            label, conf = self._argmax(probs_smooth)

            # Se não há pesos reais e a confiança é baixa, devolvemos unknown
            if not self.has_weights and conf < self.min_conf:
                return "unknown", float(conf), probs_smooth

            # Threshold opcional de confiança
            if conf < self.min_conf:
                return "unknown", float(conf), probs_smooth

            return label, float(conf), probs_smooth

        except Exception as e:
            if self.verbose:
                print(f"[EmotionNet] Falha na inferência: {e}")
            return "unknown", 0.0, {}

    # -------------------------------------------------------------------------
    # Internos: modelo, pré-processamento, suavização e utilitários
    # -------------------------------------------------------------------------

    def _build_model(self, arch: str) -> nn.Module:
        """
        Constrói a ResNet e ajusta o cabeçote para 7 classes.
        """
        arch = arch.lower()
        if arch == "resnet18":
            backbone = models.resnet18(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, self.num_classes)
            return backbone
        elif arch == "resnet34":
            backbone = models.resnet34(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, self.num_classes)
            return backbone
        elif arch == "resnet50":
            backbone = models.resnet50(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, self.num_classes)
            return backbone
        else:
            raise ValueError(f"[EmotionNet] Arquitetura não suportada: {arch}")

    def _warmup(self) -> None:
        """
        Warmup opcional (evita pico no 1º frame).
        """
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
        t = self.tf(dummy).unsqueeze(0).to(self.device)
        _ = self.model(t)

    @staticmethod
    def _equalize_lighting(img_bgr: np.ndarray) -> np.ndarray:
        """
        Equaliza iluminação no canal Y (YCrCb).
        """
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def _expand_bbox(shape: Tuple[int, int, int],
                     bbox: Tuple[int, int, int, int],
                     margin: int) -> Tuple[int, int, int, int]:
        """
        Expande o bbox com margem, respeitando limites do frame.
        """
        H, W = shape[0], shape[1]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(W, x2 + margin)
        y2 = min(H, y2 + margin)
        return x1, y1, x2, y2

    def _smooth_probs(self, probs: Dict[str, float]) -> Dict[str, float]:
        """
        Suavização temporal por média móvel (janela `self.smooth_window`).
        """
        self.smooth_q.append(probs)
        if len(self.smooth_q) == 1:
            return probs

        acc = {k: 0.0 for k in self.labels}
        for d in self.smooth_q:
            for k in self.labels:
                acc[k] += float(d.get(k, 0.0))

        cnt = float(len(self.smooth_q))
        return {k: acc[k] / cnt for k in self.labels}

    def _argmax(self, probs: Dict[str, float]) -> Tuple[str, float]:
        """
        Retorna (label, prob) da classe de maior probabilidade.
        """
        if not probs:
            return "unknown", 0.0
        label = max(probs, key=probs.get)
        return label, float(probs[label])
