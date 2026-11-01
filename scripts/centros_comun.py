"""
Parts of the centre-based detector that do not need PyTorch: preparing the input and the training
targets, turning the network's maps into boxes, and running the exported network (ONNX) with onnxruntime.
To use the detector all you need is: pip install onnxruntime
"""
import math
from pathlib import Path
import numpy as np

VECINAS = 2                     # images on each side
CIN = 2 * VECINAS + 1 + 2       # images + 2 position channels
UMBRAL_CENTRO = 0.5             # minimum peak of the centre map to propose a box (chosen on validation)
UMBRAL_DUDOSA = 0.6             # boxes with lower confidence: shown in yellow in [8] and the image goes to the list in [9] (≈ 89 % are correct)
ANCHO = 32                      # fixed width of the channel crop (centred)
LARGO = 288                     # fixed length (zero-padded; channels are up to 283 px long)
MODELO_ONNX = Path(__file__).resolve().parent / 'modelos' / 'detector_centros.onnx'


def recorte_fijo(T3):
    """(T, L, W) -> (T, LARGO, ANCHO) uint8, centred across the width and padded at the bottom."""
    T, L, W = T3.shape
    out = np.zeros((T, LARGO, ANCHO), np.uint8)
    if W >= ANCHO:
        x0 = (W - ANCHO) // 2
        out[:, :min(L, LARGO), :] = T3[:, :LARGO, x0:x0 + ANCHO]
    else:
        x0 = (ANCHO - W) // 2
        out[:, :min(L, LARGO), x0:x0 + W] = T3[:, :LARGO]
    return out


def entrada(tiles, t, fila_fondo, fila_salida):
    """tiles (T, L, W) uint8 of one channel -> (CIN, L, W) float32 for image t (0-based)."""
    T, L, W = tiles.shape
    xs = []
    for dt in range(-VECINAS, VECINAS + 1):
        u = min(max(t + dt, 0), T - 1)
        im = tiles[u].astype(np.float32)
        v = im[im > 0] if (im > 0).any() else im.ravel()
        xs.append((im - v.mean()) / max(float(v.std()), 1.0))   # blank or saturated image: do not divide by ~0
    r = np.arange(L, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    xs.append((r - fila_fondo) / 100.0); xs.append((r - fila_salida) / 100.0)
    return np.stack(xs).astype(np.float32)


def objetivos(cajas, L):
    """Centre map, distances to the poles and their mask, from your boxes [(fila_sup, fila_inf)]."""
    centro = np.zeros(L, np.float32); dist = np.zeros((2, L), np.float32); md = np.zeros(L, np.float32)
    rc = np.arange(L) + 0.5                       # centre of each row (edge coordinates, as in ImageJ)
    for a, b in cajas:
        c = 0.5 * (a + b); n = max(b - a, 1.0); s = max(1.5, n / 6)
        centro = np.maximum(centro, np.exp(-0.5 * ((rc - c) / s) ** 2))
        k = np.abs(rc - c) <= max(1.0, n / 4)
        dist[0, k] = rc[k] - a; dist[1, k] = b - rc[k]; md[k] = 1.0
    for a, b in cajas:                            # the exact peak = 1
        i = int(min(max(math.floor(0.5 * (a + b)), 0), L - 1)); centro[i] = 1.0
    return centro, dist, md


def _pico(q, lo, hi):
    """(Sub-pixel) maximum of q between rows lo and hi, and its value."""
    L = len(q); lo, hi = max(int(np.floor(lo)), 0), min(int(np.ceil(hi)) + 1, L)
    if hi <= lo: return np.nan, 0.0
    k = lo + int(np.argmax(q[lo:hi])); v = float(q[k])
    if 0 < k < L - 1:
        a, b, d = q[k - 1], q[k], q[k + 1]; den = a - 2 * b + d
        return k + (0.5 * (a - d) / den if den < 0 else 0.0), v
    return float(k), v


def afinar(cajas, P, rad=3, minimo=0.2):
    """Refine the edges with the pole probability (class output): the top pole of a box (fila_sup) is
    half a pixel above the maximum of P(upper pole), and the bottom pole half a pixel below the maximum
    of P(lower pole)."""
    out = []
    for a, b, c in cajas:
        s, vs = _pico(P[:, 2], a + 0.5 - rad, a + 0.5 + rad)
        i, vi = _pico(P[:, 3], b - 0.5 - rad, b - 0.5 + rad)
        a2 = s - 0.5 if vs >= minimo else a; b2 = i + 0.5 if vi >= minimo else b
        out.append((a2, b2, c) if b2 - a2 >= 3 else (a, b, c))
    return out


def cajas_de(centro, dist, umbral=UMBRAL_CENTRO, L=None, min_len=5.0, P=None):
    """Peaks of the centre map -> non-overlapping boxes [(fila_sup, fila_inf, confianza)].
    P (L, 4): class probabilities; if given, the edges are refined with the pole probability."""
    h = np.asarray(centro); L = L or len(h)
    h = h[:L]; d = np.asarray(dist)[:, :L]
    picos = [r for r in range(L) if h[r] >= umbral and h[r] == h[max(0, r - 3):r + 4].max()]
    cands = []
    for r in picos:
        lo, hi = max(0, r - 1), min(L, r + 2); ww = h[lo:hi]
        rc = np.arange(lo, hi) + 0.5
        a = float(np.sum(ww * (rc - d[0, lo:hi])) / ww.sum()); b = float(np.sum(ww * (rc + d[1, lo:hi])) / ww.sum())
        if b - a >= min_len: cands.append((a, b, float(h[r])))
    cands.sort(key=lambda x: -x[2]); out = []
    for a, b, c in cands:                         # 1D NMS
        if any(min(b, b2) - max(a, a2) > 0.5 * min(b - a, b2 - a2) for a2, b2, _ in out): continue
        out.append((a, b, c))
    out.sort()
    if P is not None: out = afinar(out, np.asarray(P)[:L])
    for i in range(1, len(out)):                  # small overlaps: put the edge halfway
        a0, b0, c0 = out[i - 1]; a1, b1, c1 = out[i]
        if b0 > a1:
            m = 0.5 * (b0 + a1); out[i - 1] = (a0, m, c0); out[i] = (m, b1, c1)
    return out


class DetectorCentros:
    """The network exported to ONNX, run with onnxruntime (no PyTorch)."""
    def __init__(self, ruta=MODELO_ONNX):
        import onnxruntime as ort
        op = ort.SessionOptions(); op.log_severity_level = 3
        self.s = ort.InferenceSession(str(ruta), op, providers=['CPUExecutionProvider'])

    def mapas(self, tiles, L, fila_fondo, fila_salida, lote=32):
        """tiles (T, LARGO, ANCHO) uint8 -> centre (T, L), distances (T, 2, L), classes (T, L, 4)."""
        T = tiles.shape[0]; C, D, P = [], [], []
        for i in range(0, T, lote):
            x = np.stack([entrada(tiles, t, fila_fondo, fila_salida) for t in range(i, min(T, i + lote))])
            c, d, p = self.s.run(None, {'x': x})
            C.append(c[:, :L]); D.append(d[:, :, :L]); P.append(p[:, :L])
        return np.concatenate(C), np.concatenate(D), np.concatenate(P)

    def cajas(self, tiles, L, fila_fondo, fila_salida, umbral=UMBRAL_CENTRO):
        """Boxes of each image: list (one per image) of [(fila_sup, fila_inf, confianza)], in the
        coordinates of your boxes (ImageJ coordinates in the straightened channel)."""
        C, D, P = self.mapas(tiles, L, fila_fondo, fila_salida)
        return [cajas_de(C[t], D[t], umbral, L=L, P=P[t]) for t in range(len(C))], (C, D, P)
