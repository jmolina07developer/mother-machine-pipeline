"""
Straightened channels for drawing boxes (one per cell) with the macro Cajas_Canales.ijm.

For each aligned stack it creates <stack folder>/Canales/<base>_canales_BF.tif:
  - one slice per channel (the same channels and the same numbering as the kymographs);
  - in each slice, all the frames of the channel side by side (from left to right,
    frame 1, 2, 3…), each with the straightened channel: the closed end at the top and the exit
    at the bottom, separated by a black strip;
  - the rows are the same as those of the kymograph (row 0 = 10 px above the closed end),
    so the boxes can be compared directly with the kymograph traces;
  - overlay: closed end and exit of the channel in yellow and the number of each frame at the top.
And <base>_canales_BF_geometria.csv with what is needed to map the boxes back to the original stack.

Usage:
    python canales_enderezados.py ../October/2/Stacks/01-02102026-BF.tif [more stacks…]
    (if the stack's kymographs do not exist, or are older than the stack, it first creates them with kimogramas.py)

Requires: numpy pandas scipy tifffile roifile
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kimogramas  # noqa: E402

SEP = 3            # px of black strip between frames
FRAC_ANCHO = 0.8   # crop width relative to the channel period (the walls are visible)


def recortes(stack, canal, ancho, margen_sup, largo):
    """Straightened channel in each frame: (T, largo, ancho). Same sampling as kimogramas.kimograma."""
    p0 = np.array([canal.x_sup, canal.y_sup])
    p1 = np.array([canal.x_inf, canal.y_inf])
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-u[1], u[0]])
    s = np.arange(largo) - margen_sup
    w = np.arange(ancho) - (ancho - 1) / 2
    pts = p0[None, None, :] + s[:, None, None] * u + w[None, :, None] * n
    fil, col = pts[..., 1].ravel(), pts[..., 0].ravel()
    return np.stack([map_coordinates(img.astype(np.float32), [fil, col], order=1, mode='nearest').reshape(largo, ancho)
                     for img in stack])


def fotos_sin_datos(ruta_stack, canales, margen_sup, ancho, forma):
    """
    (channel, frame) pairs in which the channel falls, wholly or partly, outside the real image of that frame: after
    a large stage jump the aligned slice is shifted and the empty part is filled with grey (Ily 03/11, positions
    07–13: a third of the field had no image until 13:00). Those frames are treated as bad frames in that channel (the
    tracking skips them; see leer_cajas.fotos_malas). Uses the shifts in <base>_registro.csv; without that file
    nothing is marked. A channel outside the image in every frame is not marked (an edge channel, as before).
    """
    base = Path(ruta_stack).stem.replace('-BF', '')
    reg = Path(ruta_stack).parent / f'{base}_registro.csv'
    if not reg.exists():
        return pd.DataFrame(columns=['canal', 'foto'])
    r = pd.read_csv(reg)
    if not {'slice', 'dy_px', 'dx_px'} <= set(r.columns):
        return pd.DataFrame(columns=['canal', 'foto'])
    H, W = forma
    filas = []
    for _, c in canales.iterrows():
        p0 = np.array([c.x_sup, c.y_sup]); p1 = np.array([c.x_inf, c.y_inf])
        u = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-u[1], u[0]])
        esquinas = np.array([p0 + s * u + w * n for s in (-margen_sup, float(c.fila_salida) - margen_sup)
                             for w in (-ancho / 4, ancho / 4)])          # the central half (the cells, not the walls)
        for t in r.itertuples():
            dy, dx = int(t.dy_px), int(t.dx_px)
            x0, x1, y0, y1 = max(0, dx), W + min(0, dx), max(0, dy), H + min(0, dy)
            dentro = (esquinas[:, 0] >= x0) & (esquinas[:, 0] <= x1 - 1) & (esquinas[:, 1] >= y0) & (esquinas[:, 1] <= y1 - 1)
            if not dentro.all():
                filas.append(dict(canal=int(c.canal), foto=int(t.slice)))
    sd = pd.DataFrame(filas, columns=['canal', 'foto'])
    n = sd.groupby('canal').size()
    return sd[~sd.canal.isin(n.index[n >= len(r)])]


def _a_8bits(R):
    """Each frame normalised to its own median (removes the flicker between frames) and converted to 8 bits."""
    R = R / (np.median(R, axis=(1, 2), keepdims=True) + 1e-6)
    lo, hi = np.percentile(R[:, :, ::3], (0.5, 99.8))
    return np.clip((R - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _superposicion(canales, T, ancho, largo):
    import roifile
    rois, paso = [], ancho + SEP
    W = T * paso
    for i, c in canales.reset_index(drop=True).iterrows():
        pos = i + 1
        for fila, nombre in ((c.fila_fondo, 'fondo'), (c.fila_salida, 'salida')):
            r = roifile.ImagejRoi.frompoints(np.array([[0, fila + 0.5], [W, fila + 0.5]], dtype=np.float32))
            r.roitype = roifile.ROI_TYPE.LINE
            r.x1, r.y1, r.x2, r.y2 = 0.0, fila + 0.5, float(W), fila + 0.5
            r.position, r.stroke_color, r.name = pos, b'\xff\xed\xa1\x00', nombre
            rois.append(r.tobytes())
        for t in range(T):                                     # frame number, at the top
            x0 = t * paso
            r = roifile.ImagejRoi(roitype=roifile.ROI_TYPE.RECT, subtype=roifile.ROI_SUBTYPE.TEXT,
                                  left=x0 + 1, top=0, right=x0 + ancho, bottom=9, text=str(t + 1),
                                  text_size=8, text_name='SansSerif', position=pos)
            r.stroke_color = b'\xff\xff\xff\xff'
            r.name = f'foto {t + 1}'
            rois.append(r.tobytes())
    return rois


def escribir_txt_macro(geo, ruta):
    """Geometry as plain text for Cajas_Canales.ijm (width, separation, number of frames and the exit of each channel)."""
    g = geo.iloc[0]
    lineas = [f'ancho {int(g.ancho)}', f'sep {int(g.sep)}', f'fotos {int(g.fotos)}']
    lineas += [f'salida {int(c)} {fs:.2f}' for c, fs in zip(geo.canal, geo.fila_salida)]
    Path(ruta).write_text('\n'.join(lineas) + '\n')


def canales_enderezados(ruta_stack):
    ruta_stack = Path(ruta_stack)
    base = ruta_stack.stem.replace('-BF', '')
    ruta_npz = ruta_stack.parent / 'Kimogramas' / f'{base}_kimogramas.npz'
    if not ruta_npz.exists() or ruta_npz.stat().st_mtime < ruta_stack.stat().st_mtime:   # stack remade: new kymographs
        kimogramas.generar_kimogramas(ruta_stack)
    z = np.load(ruta_npz, allow_pickle=True)
    canales = pd.DataFrame(z['canales'])
    margen, periodo, largo = int(z['margen_sup']), float(z['periodo']), int(z['BF'].shape[1])
    ancho = int(FRAC_ANCHO * periodo) | 1
    with tifffile.TiffFile(ruta_stack) as tif:
        st = tif.asarray()
    T = len(st)
    paso = ancho + SEP
    salida = ruta_stack.parent / 'Canales'
    salida.mkdir(exist_ok=True)
    img = np.zeros((len(canales), largo, T * paso), np.uint8)
    for k, (_, c) in enumerate(canales.iterrows()):
        R = _a_8bits(recortes(st, c, ancho, margen, largo))
        for t in range(T):
            img[k, :, t * paso:t * paso + ancho] = R[t]
    etiquetas = []
    for _, c in canales.iterrows():
        u = str(c.get('canal_usuario', '') or '')
        etiquetas.append(f'canal {int(c.canal):02d}' + (f' (usuario {u})' if u else '')
                         + f' | fondo {c.fila_fondo:.1f} | salida {c.fila_salida:.1f} | ancho {ancho} | sep {SEP} | fotos {T}')
    destino = salida / f'{base}_canales_BF.tif'
    tifffile.imwrite(destino, img, imagej=True,
                     metadata={'axes': 'ZYX', 'Labels': etiquetas, 'Overlays': _superposicion(canales, T, ancho, largo)})
    geo = canales.copy()
    geo['margen_sup'], geo['ancho'], geo['sep'], geo['largo'], geo['fotos'] = margen, ancho, SEP, largo, T
    geo['stack'] = ruta_stack.name
    geo.to_csv(salida / f'{base}_canales_BF_geometria.csv', index=False)
    escribir_txt_macro(geo, salida / f'{base}_canales_BF_macro.txt')
    usados = canales[~canales['al_borde'].astype(bool)] if 'al_borde' in canales else canales
    sd = fotos_sin_datos(ruta_stack, usados, margen, ancho, st.shape[1:])
    sd.to_csv(salida / f'{base}_canales_BF_sin_datos.csv', index=False)
    print(f'{destino.name}: {len(canales)} channels × {T} frames (each frame {ancho} × {largo} px)'
          + (f'; {len(sd)} channel-frames outside the image after a stage jump (skipped)' if len(sd) else ''))
    return destino


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    p = argparse.ArgumentParser(description='Straightened channels for drawing boxes in Fiji.')
    p.add_argument('stacks', nargs='+', help='aligned bright-field stacks (-BF.tif or _alineado.tif)')
    for s in p.parse_args().stacks:
        canales_enderezados(s)
