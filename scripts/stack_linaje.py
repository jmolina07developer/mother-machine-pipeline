"""
Aligned stack with one box per cell (automatic lineage): each cell has its own colour and number
(the number is `celula` in <imagen>_linaje.csv and _linaje_ciclos.csv, unique within each channel).
The daughters of a division get a new number and colour. The box of a cell whose lower edge is not
tracked (the last one in the channel) is drawn dashed, down to where the cell is visible or to the exit.
Only visible cells are drawn (column `visible` of _linaje.csv: interior that looks like a cell; excluding
empty channels and those cut by the image border). Images in which the network did not see a cell that
tracking knows was there (column `interpolada`) get the box filled in, dotted and with its number. Each
channel's number (from column `canal`) is drawn above it. (Loose cells from earlier versions, `seguida` = False, are drawn in grey with no number.)

Usage:
    python stack_linaje.py <stack_alineado.tif> [--canales <carpeta Canales>] [--salida <carpeta>]
                           [--fotos a-b]

Writes <stack>_linaje.tif (RGB, one image per slice), which opens in Fiji. Each image has its number and
its label (the time) written in the top-left corner. By default the TIFF is losslessly compressed
(COMPRIMIR): ImageJ stacks do not support compression, so in Fiji open it with File > Import > Bio-Formats.
"""
import argparse
import re
import sys
import colorsys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont

ANCHO_CAJA = 0.65      # box width relative to the width of the straightened channel crop
TAM_LETRA = 13         # cell numbers
TAM_CANAL = 16         # channel numbers, above each channel
COMPRIMIR = True       # losslessly compressed TIFF (zlib): ~2.7 times less space. Open it in Fiji with
                       # File > Import > Bio-Formats (or by dragging it). False: uncompressed ImageJ TIFF, as before


def color(canal, celula):
    """Stable, distinct colour for each cell (hue by golden ratio)."""
    h = ((canal * 7919 + celula) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(255 * r), int(255 * g), int(255 * b)


def hora_de(etiqueta):
    """'01_261009_141156_Regrowth_BF_00300_.tif' -> '14:11:56' (otherwise, the label as is)."""
    e = str(etiqueta).splitlines()[0]
    m = re.search(r'_\d{6}_(\d{2})(\d{2})(\d{2})_', e)
    return f'{m.group(1)}:{m.group(2)}:{m.group(3)}' if m else e


def a_8bits(st):
    lo, hi = np.percentile(st[::max(1, len(st) // 20)], [0.5, 99.5])
    return lo, max(hi, lo + 1)


def esquinas(g, sup, inf):
    """Corners of the box (rows sup–inf of the straightened crop) in aligned-stack coordinates."""
    p0 = np.array([g.x_sup, g.y_sup]); p1 = np.array([g.x_inf, g.y_inf])
    u = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-u[1], u[0]])
    m = int(g.margen_sup); w = ANCHO_CAJA * int(g.ancho) / 2
    a = p0 + (sup - 0.5 - m) * u; b = p0 + (inf - 0.5 - m) * u
    return [tuple(a - w * n), tuple(a + w * n), tuple(b + w * n), tuple(b - w * n)], (a + b) / 2 + (w + 2) * n


def discontinua(d, p, q, col, paso=4):
    p, q = np.array(p), np.array(q); L = np.linalg.norm(q - p)
    for s in np.arange(0, L, 2 * paso):
        a = p + (q - p) * s / L; b = p + (q - p) * min(s + paso, L) / L
        d.line([tuple(a), tuple(b)], fill=col, width=1)


def stack_linaje(stack, canales=None, salida=None, fotos=None):
    """
    stack   : aligned bright-field stack (…_alineado.tif or …-BF.tif)
    canales : folder with <base>_canales_BF_geometria.csv and _linaje.csv (default <stack folder>/Canales)
    salida  : folder where <stack>_linaje.tif is saved (default: the stack's folder)
    fotos   : 'a-b' to process only those images
    """
    ruta = Path(stack)
    canales = Path(canales) if canales else ruta.parent / 'Canales'
    base = ruta.stem.replace('-BF', '')                  # 01-09102026-BF.tif -> 01-09102026_canales_BF_…
    geo = pd.read_csv(canales / f'{base}_canales_BF_geometria.csv').set_index('canal')
    F = pd.read_csv(canales / f'{base}_canales_BF_linaje.csv')
    salida = Path(salida) if salida else ruta.parent
    salida.mkdir(parents=True, exist_ok=True)
    destino = salida / f'{ruta.stem}_linaje.tif'
    with tifffile.TiffFile(ruta) as tf:
        st = tf.asarray(); etiquetas = (tf.imagej_metadata or {}).get('Labels')
    T = st.shape[0]
    f0, f1 = (1, T) if not fotos else map(int, fotos.split('-'))
    lo, hi = a_8bits(st)
    try:
        letra = ImageFont.load_default(size=TAM_LETRA); letra_canal = ImageFont.load_default(size=TAM_CANAL)
    except TypeError:
        letra = letra_canal = ImageFont.load_default()
    if 'visible' in F:                                   # only the cells that are really visible (detectar_celulas.visibilidad_linaje)
        F = F[F.visible.astype(bool)]
    porfoto = {f: g for f, g in F.groupby('foto')}
    Y, X = st.shape[1:3]
    rotulos = []                                         # number of each channel, above its top end
    for c, g in geo.iterrows():
        p0 = np.array([g.x_sup, g.y_sup]); p1 = np.array([g.x_inf, g.y_inf])
        u = (p1 - p0) / np.linalg.norm(p1 - p0); e0 = p0 - int(g.margen_sup) * u
        e1 = p0 + (float(g.fila_salida) - int(g.margen_sup)) * u
        arriba = e0 if e0[1] <= e1[1] else e1
        rotulos.append(((float(np.clip(arriba[0], 8, X - 8)), float(max(arriba[1] - 3, TAM_CANAL + 1))), str(int(c))))

    def fotos():
        for f in range(f0, f1 + 1):
            im = np.clip((st[f - 1].astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(im).convert('RGB'); d = ImageDraw.Draw(img)
            for r in porfoto.get(f, pd.DataFrame()).itertuples():
                if r.canal not in geo.index: continue
                g = geo.loc[r.canal]; col = color(int(r.canal), int(r.celula))
                inf = r.fila_inf
                if r.sale_del_canal:                      # down to where the visible cell ends, or the exit
                    inf = getattr(r, 'fila_inf_visible', min(r.fila_inf, g.fila_salida + 1))
                pts, pt = esquinas(g, r.fila_sup, inf)
                if not getattr(r, 'seguida', True):     # loose cell (seen by the network, not in the lineage): grey, dashed, no number
                    col = (200, 200, 200)
                    for i in range(4): discontinua(d, pts[i], pts[(i + 1) % 4], col, paso=2)
                    continue
                if bool(getattr(r, 'interpolada', False)):  # image in which the network did not see it: filled-in, dotted box
                    for i in range(4): discontinua(d, pts[i], pts[(i + 1) % 4], col, paso=2)
                    d.text((pt[0], pt[1]), str(int(r.celula)), fill=col, font=letra, anchor='rm', stroke_width=1, stroke_fill=(0, 0, 0))
                    continue
                if r.sale_del_canal:                      # exits the channel: open, dashed box
                    d.line([pts[0], pts[1]], fill=col, width=1)
                    discontinua(d, pts[1], pts[2], col); discontinua(d, pts[3], pts[0], col)
                else:
                    d.polygon(pts, outline=col)
                d.text((pt[0], pt[1]), str(int(r.celula)), fill=col, font=letra, anchor='rm', stroke_width=1, stroke_fill=(0, 0, 0))
            for xy, txt in rotulos:
                d.text(xy, txt, fill=(255, 255, 255), font=letra_canal, anchor='mb', stroke_width=2, stroke_fill=(0, 0, 0))
            rot = f'foto {f}' + (f' · {hora_de(etiquetas[f - 1])}' if etiquetas and f - 1 < len(etiquetas) else '')
            d.text((6, 4), rot, fill=(255, 255, 255), font=letra_canal, anchor='la', stroke_width=2, stroke_fill=(0, 0, 0))
            yield np.asarray(img)

    forma = dict(shape=(f1 - f0 + 1, Y, X, 3), dtype='uint8', photometric='rgb')
    if COMPRIMIR:
        tifffile.imwrite(destino, fotos(), **forma, compression='zlib', compressionargs={'level': 6},
                         metadata={'axes': 'TYXS'})
    else:
        meta = {'axes': 'TYXS'}
        if etiquetas: meta['Labels'] = [str(e).split('\n')[0] for e in etiquetas[f0 - 1:f1]]
        tifffile.imwrite(destino, fotos(), **forma, imagej=True, metadata=meta)
    print(f'{destino.name}: {f1 - f0 + 1} images, {F.groupby(["canal", "celula"]).ngroups} cells')
    return destino


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stack')
    ap.add_argument('--canales', default=None)
    ap.add_argument('--salida', default=None)
    ap.add_argument('--fotos', default=None)
    a = ap.parse_args()
    stack_linaje(a.stack, a.canales, a.salida, a.fotos)


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    main()
