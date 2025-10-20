# mpfs/stabilizer.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple

def similarity_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna (R, t) tal que: x' = R @ x + t, com R = s*U (rotação+escala).
    src, dst: (N,2) pontos correspondentes.
    """
    src = src.astype(np.float32); dst = dst.astype(np.float32)
    mu_s = src.mean(axis=0, keepdims=True)
    mu_d = dst.mean(axis=0, keepdims=True)
    X = src - mu_s; Y = dst - mu_d
    # SVD em Y^T X
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R_ = U @ Vt
    # corrigir reflexão
    if np.linalg.det(R_) < 0:
        Vt[1, :] *= -1
        R_ = U @ Vt
    # escala
    scale = (S.sum() / (X**2).sum()) if (X**2).sum() > 1e-8 else 1.0
    R = scale * R_
    t = (mu_d.T - R @ mu_s.T).reshape(2,)
    return R, t

class LandmarkStabilizer:
    """
    Estabiliza landmarks por semelhança em relação a um “molde” (primeiro frame estável).
    """
    def __init__(self, ref_pts2d: np.ndarray | None = None, norm_pair: Tuple[int,int]=(10,152)):
        self.ref = ref_pts2d  # (N,2) no sistema de referência
        self.norm_pair = norm_pair  # para normalização métrica (altura da face)
        self.scale_ref = None

    def set_reference(self, pts2d: np.ndarray) -> None:
        self.ref = pts2d.copy()
        self.scale_ref = self._pair_dist(self.ref, *self.norm_pair)

    def apply(self, pts2d: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.ref is None:
            self.set_reference(pts2d)
            return pts2d.copy(), 1.0
        R, t = similarity_transform(pts2d, self.ref)
        aligned = (R @ pts2d.T).T + t
        scale = self._pair_dist(aligned, *self.norm_pair) / (self.scale_ref + 1e-8)
        return aligned, float(scale)

    @staticmethod
    def _pair_dist(pts: np.ndarray, i: int, j: int) -> float:
        a, b = pts[i], pts[j]
        return float(np.linalg.norm(a - b) + 1e-8)
