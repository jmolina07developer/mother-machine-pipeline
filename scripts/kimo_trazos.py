"""
Converts the borders traced by hand on a kymograph (Trazar_Kimogramas.ijm,
<base>_kimo_BF_trazos.zip) into cells, and draws them over the original stack to
check that the tracing is correct.

- Each traced border is interpolated over all the slices it spans.
- In each slice, the cells are the segments between two consecutive borders of the
  same channel (1 = the top one, the mother), except the segments marked as a gap
  (empty space, key [7] in the macro), which do not count as a cell.
- Discarded channels (key [8]) are not used, either entirely or from the given slice on.
- A cell is "exiting" (saliendo) if its lower border passes the channel exit.

Saves to the Kimogramas folder:
    <base>_celulas.csv          one row per cell and slice: canal, slice, orden,
                                fila_sup, fila_inf, longitud_px, saliendo, ...
    <base>_trazos_en_stack.tif  copy of the bright-field stack (8-bit) with the cells
                                drawn on top (Fiji overlay): one colour per position
                                in the channel (red = mother) and a mark at each border

Usage:
    python kimo_trazos.py Kimogramas/01-02102026_kimo_BF_trazos.zip
    python kimo_trazos.py Kimogramas/11-02102026_kimo_BF_auto.zip   (automatic segmentation -> files *_auto_*)
    (finds <base>_kimogramas.npz in the same folder and the stack in the parent folder automatically)

Requires: pip install roifile tifffile numpy pandas
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import roifile
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leer_roiset import _leer_rois, _propiedades  # noqa: E402

PATRON_BORDE = re.compile(r'^K(\d+)_b(\d+)$')
PATRON_MUERTE = re.compile(r'^K(\d+)_m(\d+)$')
PATRON_COMPLETO = re.compile(r'^K(\d+)_COMPLETO$')
PATRON_HUECO = re.compile(r'^K(\d+)_h(\d+)$')
PATRON_DESCARTADO = re.compile(r'^K(\d+)_DESCARTADO$')
# colours (ARGB) by the cell's order in the channel: mother, 2nd, 3rd, ...
COLORES = ['#e34948', '#2a78d6', '#eda100', '#1baf7a', '#e87ba4', '#4a3aa7', '#eb6834', '#008300']


def _argb(hexcolor):
    return bytes([255]) + bytes.fromhex(hexcolor.lstrip('#'))


def _interpolar(roi):
    """Kymograph polyline -> (t, row) at each integer slice it spans (pixel centre)."""
    pts = np.asarray(roi.coordinates(), float)
    t, f = pts[:, 0] - 0.5, pts[:, 1] - 0.5          # ImageJ -> pixel centre
    orden = np.argsort(t, kind='stable')
    t, f = t[orden], f[orden]
    ts = np.arange(max(int(np.ceil(t.min())), 0), int(np.floor(t.max())) + 1)
    return ts, np.interp(ts, t, f)


def leer_todo(ruta_zip):
    """
    Returns a dict with:
      bordes    : one row per border and slice (canal, borde, slice, fila), interpolated
      muertes   : canal, muerte, slice, fila, tipo_muerte, nota
      completos : list of channels marked as complete
      huecos    : one row per gap and slice (canal, hueco, slice, fila): empty space
      descartes : {canal: desde_slice} (1 = the whole channel)
    'fila' is the position in the kymograph (pixel centre), as in _kimo_etiquetas.csv.
    """
    filas_b, filas_m, filas_h, completos, descartes = [], [], [], [], {}
    base_zip = re.sub(r'_kimo_BF_(trazos|auto)\.zip$', '', Path(ruta_zip).name)
    ajenos, origenes = {}, {}
    for roi, texto in _leer_rois(ruta_zip):
        nombre = roi.name or ''
        props = _propiedades(texto)
        k = props.get('kimograma', '')
        if k and not k.startswith(base_zip + '_kimo'):         # traced on another kymograph
            ajenos[k] = ajenos.get(k, 0) + 1
        o = props.get('origen', 'manual')
        origenes[o] = origenes.get(o, 0) + 1
        if m := PATRON_BORDE.match(nombre):
            ts, fs = _interpolar(roi)
            filas_b += [dict(canal=int(m.group(1)), borde=int(m.group(2)), slice=int(ti) + 1, fila=float(fi))
                        for ti, fi in zip(ts, fs)]
        elif m := PATRON_HUECO.match(nombre):
            ts, fs = _interpolar(roi)
            filas_h += [dict(canal=int(m.group(1)), hueco=int(m.group(2)), slice=int(ti) + 1, fila=float(fi))
                        for ti, fi in zip(ts, fs)]
        elif m := PATRON_MUERTE.match(nombre):
            x, y = np.asarray(roi.coordinates(), float)[0]
            filas_m.append(dict(canal=int(m.group(1)), muerte=int(m.group(2)), slice=int(np.floor(x)) + 1,
                                fila=y - 0.5, tipo_muerte=props.get('tipo_muerte', ''), nota=props.get('nota', '')))
        elif m := PATRON_COMPLETO.match(nombre):
            completos.append(int(m.group(1)))
        elif m := PATRON_DESCARTADO.match(nombre):
            try:
                descartes[int(m.group(1))] = max(1, int(float(props.get('desde_slice', 1))))
            except ValueError:
                descartes[int(m.group(1))] = 1
        else:
            warnings.warn(f'ROI with unknown name (was it not saved with the macro?): "{nombre}"')
    bordes = pd.DataFrame(filas_b, columns=['canal', 'borde', 'slice', 'fila'])
    huecos = pd.DataFrame(filas_h, columns=['canal', 'hueco', 'slice', 'fila'])
    # discarded data is not used
    for c, desde in descartes.items():
        bordes = bordes[~((bordes.canal == c) & (bordes.slice >= desde))]
        huecos = huecos[~((huecos.canal == c) & (huecos.slice >= desde))]
    if ajenos:
        warnings.warn(f'{Path(ruta_zip).name}: there are ROIs traced on ANOTHER kymograph {ajenos}. '
                      'Do not mix files: each _trazos.zip must contain only the traces of its own kymograph.')
    return dict(bordes=bordes.reset_index(drop=True), muertes=pd.DataFrame(filas_m), completos=sorted(completos),
                huecos=huecos.reset_index(drop=True), descartes=descartes, ajenos=ajenos, origenes=origenes)


def leer_trazos(ruta_zip):
    """(bordes, muertes, completos), without discarded data. See leer_todo() for gaps and discards."""
    d = leer_todo(ruta_zip)
    return d['bordes'], d['muertes'], d['completos']


def celulas(bordes, canales, huecos=None):
    """
    Cells = segments between consecutive borders of a channel in each slice, except the segments
    that contain a marked gap (empty space). orden: 1 = mother, 2 = the next cell…
    hueco_encima: there is an empty space between this cell and the one above.
    """
    salida = canales.set_index('canal')['fila_salida']
    hh = {}
    if huecos is not None and len(huecos):
        for (c, s), g in huecos.groupby(['canal', 'slice']):
            hh[(c, s)] = g.fila.to_numpy()
    filas = []
    for (c, s), g in bordes.groupby(['canal', 'slice']):
        g = g.sort_values('fila')
        f, b = g.fila.to_numpy(), g.borde.to_numpy()
        hs = hh.get((c, s), np.array([]))
        orden, hueco_antes = 0, False
        for k in range(len(g) - 1):
            if np.any((hs > f[k]) & (hs < f[k + 1])):
                hueco_antes = True
                continue
            orden += 1
            filas.append(dict(canal=c, slice=s, orden=orden, borde_sup=b[k], borde_inf=b[k + 1],
                              fila_sup=f[k], fila_inf=f[k + 1], longitud_px=f[k + 1] - f[k],
                              hueco_encima=hueco_antes, saliendo=bool(f[k + 1] > salida.get(c, np.inf))))
            hueco_antes = False
    return pd.DataFrame(filas)


def _punto(canal, fila, margen_sup):
    """Kymograph row -> (x, y) in the stack (pixel centre), and the axis vectors."""
    p0 = np.array([canal['x_sup'], canal['y_sup']], float)
    p1 = np.array([canal['x_inf'], canal['y_inf']], float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    return p0 + (fila - margen_sup) * u, u, np.array([-u[1], u[0]])


def superposicion(cels, bordes, canales, margen_sup, periodo):
    """ROIs (bytes) for the overlay: one line per cell along the axis and one mark per border."""
    can = canales.set_index('canal')
    rois = []
    for _, r in cels.iterrows():
        a, u, n = _punto(can.loc[r.canal], r.fila_sup, margen_sup)
        b, _, _ = _punto(can.loc[r.canal], r.fila_inf, margen_sup)
        a, b = a + 0.5 + u * 1.0, b + 0.5 - u * 1.0             # +0.5: to ImageJ coordinates; gap at the borders
        roi = roifile.ImagejRoi.frompoints(np.array([a, b], dtype=np.float32))
        roi.roitype = roifile.ROI_TYPE.LINE
        roi.x1, roi.y1, roi.x2, roi.y2 = float(a[0]), float(a[1]), float(b[0]), float(b[1])
        roi.position = int(r.slice)
        roi.stroke_color = _argb(COLORES[(int(r.orden) - 1) % len(COLORES)])
        roi.stroke_width = 2
        roi.name = f'T{int(r.canal):02d}_c{int(r.orden)}_s{int(r.slice):03d}'
        rois.append(roi.tobytes())
    media = 0.3 * periodo
    for _, r in bordes.iterrows():
        p, u, n = _punto(can.loc[r.canal], r.fila, margen_sup)
        a, b = p + 0.5 - n * media, p + 0.5 + n * media
        roi = roifile.ImagejRoi.frompoints(np.array([a, b], dtype=np.float32))
        roi.roitype = roifile.ROI_TYPE.LINE
        roi.x1, roi.y1, roi.x2, roi.y2 = float(a[0]), float(a[1]), float(b[0]), float(b[1])
        roi.position = int(r.slice)
        roi.stroke_color = _argb('#ffffff')
        roi.name = f'T{int(r.canal):02d}_b{int(r.borde):03d}_s{int(r.slice):03d}'
        rois.append(roi.tobytes())
    return rois


def procesar_trazos(ruta_trazos, ruta_npz=None, ruta_stack=None, pintar=True):
    ruta_trazos = Path(ruta_trazos)
    auto = ruta_trazos.name.endswith('_kimo_BF_auto.zip')
    base = re.sub(r'_kimo_BF_(trazos|auto)\.zip$', '', ruta_trazos.name)
    suf = '_auto' if auto else ''                  # do not overwrite the results of the manual traces
    ruta_npz = Path(ruta_npz) if ruta_npz else ruta_trazos.with_name(f'{base}_kimogramas.npz')
    z = np.load(ruta_npz, allow_pickle=True)
    canales = pd.DataFrame(z['canales'])
    margen_sup, periodo = int(z['margen_sup']), float(z['periodo'])
    tiempo = z['tiempo_min']

    todo = leer_todo(ruta_trazos)
    bordes, muertes, completos = todo['bordes'], todo['muertes'], todo['completos']
    huecos, descartes = todo['huecos'], todo['descartes']
    if bordes.empty:
        raise ValueError(f'No borders in {ruta_trazos.name}')
    cels = celulas(bordes, canales, huecos)
    cels['tiempo_min'] = tiempo[cels.slice.to_numpy() - 1]
    cels['canal_completo'] = cels.canal.isin(completos)
    usuario = canales.set_index('canal').get('canal_usuario')
    if usuario is not None:
        cels['canal_usuario'] = cels.canal.map(usuario)
    cels.to_csv(ruta_trazos.with_name(f'{base}{suf}_celulas.csv'), index=False)
    if not muertes.empty:
        muertes['tiempo_min'] = tiempo[muertes.slice.to_numpy() - 1]
        muertes.to_csv(ruta_trazos.with_name(f'{base}{suf}_muertes_kimo.csv'), index=False)

    print(f'{ruta_trazos.name}: {bordes.groupby(["canal", "borde"]).ngroups} borders in {bordes.canal.nunique()} channels, '
          f'{len(cels)} cells·slice, {huecos.groupby(["canal", "hueco"]).ngroups if len(huecos) else 0} gaps, '
          f'{len(muertes)} deaths, complete channels: {completos or "none"}, '
          f'discarded: {", ".join(f"{c}" + (f" (from {d})" if d > 1 else "") for c, d in sorted(descartes.items())) or "none"}')

    if pintar:
        ruta_stack = Path(ruta_stack) if ruta_stack else ruta_trazos.parent.parent / str(z['stack'])
        with tifffile.TiffFile(ruta_stack) as tif:
            st = tif.asarray()
            etiquetas = (tif.imagej_metadata or {}).get('Labels')
        if st.dtype != np.uint8:                    # lightweight 8-bit copy, for viewing only
            lo, hi = np.percentile(st[:, ::4, ::4], (0.1, 99.9))
            st = np.clip((st.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        meta = {'axes': 'ZYX', 'Overlays': superposicion(cels, bordes, canales, margen_sup, periodo)}
        if etiquetas:
            meta['Labels'] = etiquetas
        destino = ruta_trazos.with_name(f'{base}{suf}_trazos_en_stack.tif')
        tifffile.imwrite(destino, st, imagej=True, metadata=meta)
        print(f'Saved: {destino.name} (open it in Fiji; the cells are shown as an overlay)')
    return cels, bordes, muertes


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    import argparse
    p = argparse.ArgumentParser(description='Borders traced on the kymograph -> cells, drawn over the stack.')
    p.add_argument('trazos', nargs='+', help='<base>_kimo_BF_trazos.zip or <base>_kimo_BF_auto.zip')
    p.add_argument('--sin-pintar', action='store_true', help='only the cell table')
    a = p.parse_args()
    for t in a.trazos:
        procesar_trazos(t, pintar=not a.sin_pintar)
