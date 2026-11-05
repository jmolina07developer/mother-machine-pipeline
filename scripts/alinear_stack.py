"""
Aligns a mother machine stack using the chip numbers as reference
(the strip at the top, which does not change over time).

- Computes the shift (dy, dx) of each slice relative to the first one by finding
  where the number region of the first slice appears in each slice
  (template matching with normalised correlation on the edges of the digits).
- Shifts each slice by a whole number of pixels (does not blur the image).
- Saves <stack>_alineado.tif KEEPING the slice names (the times).
- Saves <stack>_alineado_desplazamientos.csv and a .png with the shifts.
- The quality of each slice is that correlation (1 = perfect match); it warns about
  the slices with low values so they can be checked.
- The original stack is not modified.

Usage from the terminal:
    python alinear_stack.py 03-25092026-stack.tif
    python alinear_stack.py 03-25092026-stack.tif --recortar

Usage in a notebook:
    from alinear_stack import alinear_stack
    tabla = alinear_stack('03-25092026-stack.tif')

Requires: pip install tifffile scikit-image pandas matplotlib
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.feature import match_template
from skimage.filters import difference_of_gaussians

ZONA_NUMEROS = (0, 450)   # rows (y_ini, y_fin) where the chip numbers are. The search uses this region
                          # without its top and bottom margin: with (0, 350) the numbers fell outside it when they
                          # were lower down (Ily 13/10, positions 10 and 11: around row 300) and the
                          # alignment locked onto the camera's dust specks, which do not move
MARGEN = 40               # maximum shift searched (px) in each direction
Q_ANCHO = 0.93            # slices matching worse than this are also searched with large shifts
SALTO_MAX = 500           # maximum shift (px) of that wide search
SOLAPE_MIN = 0.35         # minimum fraction of the image that must overlap with the reference


def _bordes(img):
    """Enhances the edges of the digits and removes the background (difference of Gaussians)."""
    return difference_of_gaussians(img.astype(float), 2, 20)


def _desplazar(img, dy, dx, relleno):
    """Shifts img by (dy, dx) whole pixels; the area left empty is filled with `relleno`."""
    out = np.full_like(img, relleno)
    h, w = img.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def _ncc_solape(A, M, dy, dx):
    """Normalised correlation between A and M shifted by (dy, dx), only where they overlap, and the overlapping fraction."""
    h, w = A.shape
    ya, yb, xa, xb = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    if yb - ya < 10 or xb - xa < 10:
        return -1.0, 0.0
    a = A[ya:yb, xa:xb]; m = M[ya - dy:yb - dy, xa - dx:xb - dx]
    a = a - a.mean(); m = m - m.mean()
    den = np.sqrt((a * a).sum() * (m * m).sum())
    return (float((a * m).sum() / den) if den > 0 else -1.0), (yb - ya) * (xb - xa) / (h * w)


def medir_ancho(A, M, salto_max=SALTO_MAX, solape_min=SOLAPE_MIN):
    """
    Shift (dy, dx) that brings M onto A, searching ALL shifts with an overlap of at least solape_min (masked
    normalised cross-correlation in the Fourier domain, skimage), first at half resolution and then refined ±2 px.
    It handles stage jumps larger than the normal search margin (Ily 03/11, positions 07–13: ~320 px before 13:00,
    ~90 px vertically at 14:31 and 15:43), where the normal search locks onto the neighbouring block of numbers (the
    numbers repeat every ~200 px). Returns (dy, dx, quality), quality = correlation in the overlapping area.
    """
    from skimage.registration import phase_cross_correlation
    a, m = A[::2, ::2], M[::2, ::2]
    s = phase_cross_correlation(a, m, reference_mask=np.ones(a.shape, bool), moving_mask=np.ones(m.shape, bool),
                                overlap_ratio=solape_min)
    s = s[0] if isinstance(s, tuple) else s
    dy0, dx0 = int(round(2 * s[0])), int(round(2 * s[1]))
    mejor = (-1.0, 0, 0)
    for i in range(-2, 3):
        for j in range(-2, 3):
            q, f = _ncc_solape(A, M, dy0 + i, dx0 + j)
            if f >= solape_min and q > mejor[0]:
                mejor = (q, dy0 + i, dx0 + j)
    q, dy, dx = mejor
    if abs(dy) > salto_max or abs(dx) > salto_max:
        return 0, 0, -1.0
    return dy, dx, q


def calcular_desplazamientos(stack, zona=ZONA_NUMEROS, margen=MARGEN, referencia='auto', calidad_min=0.8):
    """
    Shift (dy, dx) of each slice relative to a reference slice, searching each one
    for the number region of the reference (normalised correlation on the edges).

    referencia='auto': tries as reference the slices at 0, 25, 50 and 75 % of the stack and
    picks the one that best matches the rest (so it does not fail if the first slice is unusual).
    Slices that match the reference poorly are retried against the other candidates.

    Returns a DataFrame: slice, dy_px, dx_px, calidad (1 = perfect match), comparado_con,
    salto_px. The reference slice is stored in tabla.attrs['slice_referencia'].
    """
    y0, y1 = zona
    B = [_bordes(img)[y0:y1].astype(np.float32) for img in stack]
    n = len(B)

    def medir(i, k):
        """Shift that maps slice i onto slice k, and its quality."""
        plantilla = B[k][margen:-margen, margen:-margen]
        ncc = match_template(B[i], plantilla)
        r, c = np.unravel_index(np.argmax(ncc), ncc.shape)
        return int(margen - r), int(margen - c), float(ncc[r, c])

    if referencia == 'auto':
        candidatos = sorted({0, n // 4, n // 2, (3 * n) // 4})
        muestra = np.unique(np.linspace(0, n - 1, min(n, 12)).astype(int))
        puntuacion = {k: np.median([medir(i, k)[2] for i in muestra if i != k]) for k in candidatos}
        ref = max(puntuacion, key=puntuacion.get)
    else:
        ref = int(referencia) - 1
        candidatos = [ref]

    res = [list(medir(i, ref)) + [ref + 1] for i in range(n)]

    # Rescue: slices that match the reference poorly are tried against the other candidates
    otras = {k: medir(k, ref) for k in candidatos if k != ref}
    otras = {k: o for k, o in otras.items() if o[2] >= calidad_min}
    for i in range(n):
        if res[i][2] >= calidad_min:
            continue
        for k, (oy, ox, _) in otras.items():
            dy, dx, q = medir(i, k)
            if q > res[i][2]:
                res[i] = [dy + oy, dx + ox, q, k + 1]

    # Large jumps: beyond the margin, or the search locked onto the neighbouring block of numbers (quality ~0.7–0.86
    # instead of >0.95). All shifts are searched and the best match is kept.
    for i in range(n):
        if i == ref or res[i][2] >= Q_ANCHO:
            continue
        dy, dx, q = medir_ancho(B[ref], B[i])
        if q > res[i][2] + 0.02:
            res[i] = [dy, dx, q, ref + 1]

    tabla = pd.DataFrame(res, columns=['dy_px', 'dx_px', 'calidad', 'comparado_con'])
    tabla.insert(0, 'slice', np.arange(1, n + 1))
    tabla['salto_px'] = np.hypot(tabla.dy_px.diff(), tabla.dx_px.diff()).fillna(0)
    tabla.attrs['slice_referencia'] = ref + 1
    return tabla


def comprobar_alineacion(alineado, filas=(20, 450), margen=15):
    """
    Independent check of the alignment: residual shift between each already aligned slice
    and the previous one (should be 0–1 px) and how well they match. Returns (residuo_px, encaje) per slice.
    """
    y0, y1 = filas
    B = [_bordes(img)[y0:y1].astype(np.float32) for img in alineado]
    res, enc = [0.0], [1.0]
    for i in range(1, len(B)):
        ncc = match_template(B[i - 1], B[i][margen:-margen, margen:-margen])
        r, c = np.unravel_index(np.argmax(ncc), ncc.shape)
        res.append(float(max(abs(margen - r), abs(margen - c)))); enc.append(float(ncc[r, c]))
    return np.array(res), np.array(enc)


def aplicar_desplazamientos(stack, tabla):
    """Applies to each slice of `stack` the shift given in `tabla` (columns dy_px, dx_px)."""
    alineado = np.empty_like(stack)
    for i, (dy, dx) in enumerate(zip(tabla.dy_px, tabla.dx_px)):
        relleno = np.median(stack[i]).astype(stack.dtype)
        alineado[i] = _desplazar(stack[i], int(dy), int(dx), relleno)
    return alineado


def avisos(tabla, margen=MARGEN, calidad_min=0.8, nombre=''):
    al_limite = tabla[((tabla.dy_px.abs() == margen) | (tabla.dx_px.abs() == margen)) & (tabla.calidad < Q_ANCHO)]
    if not al_limite.empty:
        warnings.warn(f'{nombre} slices {al_limite.slice.tolist()}: the shift reaches the search margin '
                      f'({margen} px); it may be larger. Try a larger margin.')
    malos = tabla[tabla.calidad < calidad_min]
    if not malos.empty:
        warnings.warn(f'{nombre} alignment quality < {calidad_min} in slices {malos.slice.tolist()}: '
                      'check them in Fiji (the image may be out of focus or the alignment may have failed)')


def alinear_stack(ruta, zona=ZONA_NUMEROS, margen=MARGEN, recortar=False, calidad_min=0.8, grafico=True):
    """
    Parameters
    ----------
    ruta        : .tif stack (ImageJ) to align
    zona        : rows (y_ini, y_fin) where the chip numbers are
    margen      : maximum shift searched (px)
    recortar    : if True, crops to the area covered by all slices (no empty borders)
    calidad_min : warns about slices whose alignment quality falls below this value
    grafico     : saves a .png with the shifts

    Returns a DataFrame with the shift and quality of each slice.
    """
    ruta = Path(ruta)
    with tifffile.TiffFile(ruta) as tif:
        stack = tif.asarray()
        meta = tif.imagej_metadata or {}
    if stack.ndim != 3:
        raise ValueError(f'Expected a 3D stack (slices, y, x) but got shape {stack.shape}')
    etiquetas = meta.get('Labels') or [''] * len(stack)

    tabla = calcular_desplazamientos(stack, zona, margen)
    tabla.insert(4, 'etiqueta_slice', [e.split('\n')[0].strip() for e in etiquetas])
    alineado = aplicar_desplazamientos(stack, tabla)

    if recortar:   # region that has real image data in every slice
        y0 = max(0, tabla.dy_px.max()); y1 = stack.shape[1] + min(0, tabla.dy_px.min())
        x0 = max(0, tabla.dx_px.max()); x1 = stack.shape[2] + min(0, tabla.dx_px.min())
        alineado = alineado[:, y0:y1, x0:x1]

    salida = ruta.with_name(ruta.stem + '_alineado.tif')
    tifffile.imwrite(salida, alineado, imagej=True,
                     metadata={'Labels': list(etiquetas), 'axes': 'ZYX'})
    tabla.to_csv(salida.with_name(salida.stem + '_desplazamientos.csv'), index=False)

    avisos(tabla, margen, calidad_min)

    if grafico:
        import matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(tabla.slice, tabla.dx_px, label='dx', color='#2a78d6', lw=1.5)
        ax.plot(tabla.slice, tabla.dy_px, label='dy', color='#eb6834', lw=1.5)
        ax.set_xlabel('Slice'); ax.set_ylabel('Corrected shift (px)')
        ax.set_title(ruta.name, loc='left', fontsize=10)
        ax.legend(frameon=False); ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(salida.with_name(salida.stem + '_desplazamientos.png'), dpi=120)
        if matplotlib.get_backend().lower() == 'agg':
            plt.close(fig)

    print(f'{ruta.name}: dx from {tabla.dx_px.min():.1f} to {tabla.dx_px.max():.1f} px, '
          f'dy from {tabla.dy_px.min():.1f} to {tabla.dy_px.max():.1f} px')
    print(f'Jumps > 3 px between consecutive slices: {(tabla.salto_px > 3).sum()} | '
          f'median quality {tabla.calidad.median():.2f} (minimum {tabla.calidad.min():.2f})')
    print(f'Saved: {salida.name} {alineado.shape}')
    return tabla


# Only from the terminal (inside Jupyter, sys.argv holds the kernel's arguments)
if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    p = argparse.ArgumentParser(description='Aligns a stack using the chip numbers.')
    p.add_argument('stack')
    p.add_argument('--recortar', action='store_true', help='crop the empty borders')
    p.add_argument('--margen', type=int, default=MARGEN, help=f'maximum shift searched (px, default {MARGEN})')
    a = p.parse_args()
    alinear_stack(a.stack, margen=a.margen, recortar=a.recortar)
