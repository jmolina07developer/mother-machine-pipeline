"""
Centre-based cell detector (object detection with boxes, no masks) for straightened channels.

For each row of the channel the network predicts:
  - centre: probability that the centre of a cell is there (heat map, as in CenterNet);
  - distance to the top pole and to the bottom pole of that cell (in px);
  - class of the row (background, cell, top pole, bottom pole), the same output as the previous
    detector, so that mother/daughter tracking and the lineage work the same way.
One box = one peak of the centre map + its two distances. Two touching cells have separate centres,
so they do not merge.

Input: the image and its neighbours (t-2 … t+2), each one normalised, plus two position channels
(distance to the closed end and to the exit). Architecture: 2D convolutions that progressively narrow
the channel width until it is collapsed, followed by a 1D U-Net along the channel.

It is trained on your boxes (proposals you deleted count as "no cell here", with a higher weight;
below your last box, background with half weight), like the previous detector.
"""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from centros_comun import (VECINAS, CIN, UMBRAL_CENTRO, UMBRAL_DUDOSA, ANCHO, LARGO, recorte_fijo, entrada,
                           objetivos, cajas_de, afinar, _pico)


# =========================================================================== network
def _b2(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


def _b1(ci, co, d=1):
    return nn.Sequential(nn.Conv1d(ci, co, 3, padding=d, dilation=d, bias=False), nn.BatchNorm1d(co), nn.ReLU(inplace=True),
                         nn.Conv1d(co, co, 3, padding=d, dilation=d, bias=False), nn.BatchNorm1d(co), nn.ReLU(inplace=True))


class RedCentros(nn.Module):
    def __init__(self, cin=CIN, a=16):
        super().__init__()
        self.e1 = _b2(cin, a)            # (L, 32)
        self.e2 = _b2(a, 2 * a)          # (L, 16)
        self.e3 = _b2(2 * a, 4 * a)      # (L, 8)
        self.col = nn.Sequential(nn.Conv2d(4 * a, 4 * a, (1, 8), bias=False), nn.BatchNorm2d(4 * a), nn.ReLU(inplace=True))
        c0, c1, c2, c3 = 4 * a, 6 * a, 8 * a, 8 * a
        self.u0 = _b1(c0, c0)
        self.d1 = _b1(c0, c1); self.d2 = _b1(c1, c2); self.d3 = nn.Sequential(_b1(c2, c3, 2), _b1(c3, c3, 4))
        self.s2 = _b1(c3 + c2, c2); self.s1 = _b1(c2 + c1, c1); self.s0 = _b1(c1 + c0, c0)
        self.centro = nn.Conv1d(c0, 1, 1); self.dist = nn.Conv1d(c0, 2, 1); self.clase = nn.Conv1d(c0, 4, 1)
        nn.init.constant_(self.centro.bias, -2.2)

    def forward(self, x):                 # x: (B, CIN, L, 32)
        x = self.e1(x); x = Fn.max_pool2d(x, (1, 2))
        x = self.e2(x); x = Fn.max_pool2d(x, (1, 2))
        x = self.e3(x); x = self.col(x).squeeze(-1)          # (B, C, L)
        x0 = self.u0(x)
        x1 = self.d1(Fn.max_pool1d(x0, 2)); x2 = self.d2(Fn.max_pool1d(x1, 2)); x3 = self.d3(Fn.max_pool1d(x2, 2))
        y = self.s2(torch.cat([Fn.interpolate(x3, size=x2.shape[-1]), x2], 1))
        y = self.s1(torch.cat([Fn.interpolate(y, size=x1.shape[-1]), x1], 1))
        y = self.s0(torch.cat([Fn.interpolate(y, size=x0.shape[-1]), x0], 1))
        return self.centro(y)[:, 0], Fn.softplus(self.dist(y)) * 10.0, self.clase(y)


# =========================================================================== input


# =========================================================================== targets


# =========================================================================== loss
def perdida(salida, obj, pesos_clase):
    z, d, cl = salida
    centro, dist, md, y, w = obj
    p = torch.sigmoid(z).clamp(1e-4, 1 - 1e-4)
    pos = (centro >= 0.999).float()
    neg = (1 - pos) * (1 - centro) ** 4
    focal = -(pos * (1 - p) ** 2 * torch.log(p) + neg * p ** 2 * torch.log(1 - p)) * w
    n = pos.sum().clamp(min=1)
    l_centro = focal.sum() / n
    l_dist = (md.unsqueeze(1) * (torch.log1p(d) - torch.log1p(dist)).abs()).sum() / md.sum().clamp(min=1) / 2
    ce = Fn.cross_entropy(cl, y.long(), weight=pesos_clase, reduction='none') * w
    l_clase = ce.sum() / w.sum()
    return l_centro + l_dist + 0.5 * l_clase, (l_centro.item(), l_dist.item(), l_clase.item())


# =========================================================================== boxes






class _Exportable(nn.Module):
    """For exporting to ONNX: returns probabilities (centre, distances, per-row classes)."""
    def __init__(self, red):
        super().__init__(); self.red = red
    def forward(self, x):
        z, d, cl = self.red(x)
        return torch.sigmoid(z), d, torch.softmax(cl, 1).permute(0, 2, 1)


def exportar_onnx(ruta_pt, ruta_onnx):
    ck = torch.load(ruta_pt, weights_only=False); red = RedCentros(); red.load_state_dict(ck['estado']); red.eval()
    x = torch.zeros(2, CIN, LARGO, ANCHO)
    torch.onnx.export(_Exportable(red), x, str(ruta_onnx), input_names=['x'], output_names=['centro', 'dist', 'clases'],
                      dynamic_axes={'x': {0: 'n'}, 'centro': {0: 'n'}, 'dist': {0: 'n'}, 'clases': {0: 'n'}},
                      opset_version=17, dynamo=False)


@torch.no_grad()
def predecir_canal(red, tiles, L, fila_fondo, fila_salida, lote=32):
    """All images of one channel: centre (T, L), distances (T, 2, L), class probabilities (T, L, 4)."""
    red.eval(); T = tiles.shape[0]; C, D, P = [], [], []
    for i in range(0, T, lote):
        x = torch.from_numpy(np.stack([entrada(tiles, t, fila_fondo, fila_salida) for t in range(i, min(T, i + lote))]))
        z, d, cl = red(x)
        C.append(torch.sigmoid(z)[:, :L].numpy()); D.append(d[:, :, :L].numpy())
        P.append(torch.softmax(cl, 1)[:, :, :L].permute(0, 2, 1).numpy())
    return np.concatenate(C), np.concatenate(D), np.concatenate(P)
