"""
Reads the boxes drawn with Cajas_Canales.ijm (<base>_canales_BF_cajas.zip) and turns them into a
table, one row per cell and frame:

    canal, foto, caja, origen (manual, copiada or detector), orden (1 = the top one), fila_sup, fila_inf (kymograph rows,
    pixel centre), longitud_px, x_izq, x_der (within the frame, 0 = left edge of the
    crop), sale_del_canal, longitud_valida_px

- sale_del_canal: the lower pole of the box passes the channel exit (fila_salida - 1).
  These boxes are kept (they say where the cell is) but longitud_valida_px is left empty,
  so that they are not used when measuring lengths.
- fotos_vacias: (canal, foto) marked with [6] (channel without cells).
- leer_borradas(): boxes proposed by the detector ([8]) that you deleted with [2] (the macro
  records them in <imagen>_borradas.txt): there is no cell there. They are used for training.
- fotos_malas(): (canal, foto) marked with [6] as a defective frame (out of focus, shifted…);
  canal 0 = in all channels. They do not count as annotated and tracking skips them.
- anotadas: (canal, foto) with at least one box or marked as empty. The other frames were not
  annotated and are not used.

Usage:
    python leer_cajas.py ../October/2/Stacks/Canales/11-02102026_canales_BF_cajas.zip
    -> saves <base>_cajas.csv next to the zip

Requires: numpy pandas roifile
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leer_roiset import _leer_rois, _propiedades  # noqa: E402

PATRON_CAJA = re.compile(r'^C(\d+)_f(\d+)_c(\d+)$')
PATRON_VACIA = re.compile(r'^C(\d+)_f(\d+)_VACIA$')
PATRON_MALA = re.compile(r'^C(\d+)_f(\d+)_MALA$')


def leer_cajas(ruta_zip, ruta_geometria=None):
    """Returns (cajas, fotos_vacias, anotadas)."""
    ruta_zip = Path(ruta_zip)
    base_img = ruta_zip.name.replace('_cajas.zip', '')
    ruta_geometria = Path(ruta_geometria) if ruta_geometria else ruta_zip.with_name(f'{base_img}_geometria.csv')
    geo = pd.read_csv(ruta_geometria).set_index('canal')
    ancho, sep = int(geo.ancho.iloc[0]), int(geo.sep.iloc[0])
    paso = ancho + sep
    filas, vacias, ajenos = [], [], {}
    for roi, texto in _leer_rois(ruta_zip):
        nombre = roi.name or ''
        props = _propiedades(texto)
        img = props.get('imagen', '')
        if img and not img.startswith(base_img):
            ajenos[img] = ajenos.get(img, 0) + 1
            continue
        if m := PATRON_CAJA.match(nombre):
            c, f, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pts = np.asarray(roi.coordinates(), float)
            x0, x1 = pts[:, 0].min(), pts[:, 0].max()
            y0, y1 = pts[:, 1].min(), pts[:, 1].max()
            tile = int((0.5 * (x0 + x1)) // paso)
            if tile + 1 != f:
                warnings.warn(f'{nombre}: the box is in frame {tile + 1}, not in {f}; using {tile + 1}')
                f = tile + 1
            filas.append(dict(canal=c, foto=f, caja=k, fila_sup=y0 - 0.5, fila_inf=y1 - 0.5,
                              x_izq=x0 - tile * paso, x_der=x1 - tile * paso, origen=props.get('origen', 'manual')))
        elif m := PATRON_VACIA.match(nombre):
            vacias.append(dict(canal=int(m.group(1)), foto=int(m.group(2))))
        elif PATRON_MALA.match(nombre):
            pass                                             # defective frames: fotos_malas()
        elif nombre:
            warnings.warn(f'ROI with unknown name (was it not saved with the macro?): "{nombre}"')
    if ajenos:
        warnings.warn(f'{ruta_zip.name}: there are boxes from ANOTHER image {ajenos}; they are not used.')
    cajas = pd.DataFrame(filas, columns=['canal', 'foto', 'caja', 'fila_sup', 'fila_inf', 'x_izq', 'x_der', 'origen'])
    if len(cajas):
        cajas = cajas.sort_values(['canal', 'foto', 'fila_sup']).reset_index(drop=True)
        cajas['orden'] = cajas.groupby(['canal', 'foto']).cumcount() + 1
        cajas['longitud_px'] = cajas.fila_inf - cajas.fila_sup
        cajas['fila_salida'] = cajas.canal.map(geo.fila_salida)
        cajas['sale_del_canal'] = cajas.fila_inf >= cajas.fila_salida - 1
        cajas['longitud_valida_px'] = cajas.longitud_px.where(~cajas.sale_del_canal)
        if 'canal_usuario' in geo:
            cajas['canal_usuario'] = cajas.canal.map(geo.canal_usuario)
        # large overlaps between boxes in the same frame: probably a drawing mistake
        for (c, f), g in cajas.groupby(['canal', 'foto']):
            a, b = g.fila_sup.to_numpy(), g.fila_inf.to_numpy()
            sol = np.minimum(b[:-1], b[1:]) - np.maximum(a[:-1], a[1:])
            if np.any(sol > 0.5 * np.minimum(b - a, np.roll(b - a, -1))[:-1]):
                warnings.warn(f'channel {c}, frame {f}: two boxes overlap a lot')
    vacias = pd.DataFrame(vacias, columns=['canal', 'foto'])
    anotadas = pd.concat([cajas[['canal', 'foto']], vacias]).drop_duplicates().sort_values(['canal', 'foto'])
    return cajas, vacias, anotadas.reset_index(drop=True)


def fotos_malas(ruta_zip):
    """Frames marked as defective: DataFrame (canal, foto); canal 0 = in all channels."""
    ruta_zip = Path(ruta_zip)
    base_img = ruta_zip.name.replace('_cajas.zip', '')
    filas = []
    if ruta_zip.exists():
        for roi, texto in _leer_rois(ruta_zip):
            img = _propiedades(texto).get('imagen', '')
            if img and not img.startswith(base_img): continue
            if m := PATRON_MALA.match(roi.name or ''):
                filas.append(dict(canal=int(m.group(1)), foto=int(m.group(2))))
    return pd.DataFrame(filas, columns=['canal', 'foto'])


def leer_borradas(ruta_zip):
    """Detector proposals that you deleted: DataFrame (canal, foto, y0, y1) in the same
    coordinates as <imagen>_detector.txt (y0 = fila_sup + 0.5)."""
    ruta = Path(ruta_zip).with_name(Path(ruta_zip).name.replace('_cajas.zip', '_borradas.txt'))
    filas = []
    if ruta.exists():
        for l in ruta.read_text().splitlines():
            p = l.split()
            if len(p) >= 4:
                try:
                    filas.append(dict(canal=int(p[0]), foto=int(p[1]), y0=float(p[2]), y1=float(p[3])))
                except ValueError:
                    pass
    return pd.DataFrame(filas, columns=['canal', 'foto', 'y0', 'y1']).drop_duplicates()


def malas_del_canal(malas, canal):
    """Defective frames of a channel (its own and those marked in all channels)."""
    return sorted(set(malas.foto[(malas.canal == canal) | (malas.canal == 0)].astype(int)))


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    import argparse
    p = argparse.ArgumentParser(description='Boxes from Cajas_Canales.ijm -> table.')
    p.add_argument('zips', nargs='+', help='<base>_canales_BF_cajas.zip')
    for z in p.parse_args().zips:
        z = Path(z)
        cajas, vacias, anotadas = leer_cajas(z)
        malas = fotos_malas(z)
        destino = z.with_name(z.name.replace('_cajas.zip', '_cajas.csv'))
        cajas.to_csv(destino, index=False)
        if len(malas):
            malas.to_csv(z.with_name(z.name.replace('_cajas.zip', '_fotos_malas.csv')), index=False)
        print(f'{z.name}: {len(cajas)} boxes in {len(anotadas)} annotated frames from {anotadas.canal.nunique()} channels '
              f'({int(cajas.sale_del_canal.sum()) if len(cajas) else 0} pass the exit, {len(vacias)} empty frames, '
              f'{len(malas)} defective frames) '
              f'-> {destino.name}')
