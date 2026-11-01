"""
Fluorescence (GFP, PI…) of each cell of the automatic lineage in each image.

For each position it reads the aligned fluorescence stack (<pos>-<fecha>-<FLUO>.tif, same pairing and
alignment as the bright field), straightens each channel exactly like the bright field
(canales_enderezados.recortes, same geometry) and, for each cell-image of _linaje.csv, computes:

- mean intensity inside the box: the central columns of the channel (COLUMNAS px) and the cell's
  rows minus RECORTE_POLO px at each pole;
- wall background: median intensity of the channel walls (the COLUMNAS_PARED columns on each side)
  in those same rows and that same image (the background increases towards the channel exit and
  changes over time);
- medium background: how much brighter the medium inside an empty channel is than the wall, the
  per-image median over the cell-free stretches of all channels of the position;
- fluo_neta = mean − wall − medium (≈ protein concentration, arbitrary units) and
  fluo_total = fluo_neta × length (≈ amount).

Only visible cells are measured, with a measured length or an interpolated box (fluo_neta is left empty
if the box touches the exit: the visible piece may not be representative… it is measured anyway, with
`toca_salida` = True, so that it can be filtered out).
Black slices (no image for that fluorescence: SIN_<FLUO> in the stack) are not measured.

Writes <base>_canales_BF_linaje_<FLUO>.csv next to _linaje.csv (canal, foto, celula, fluo_media,
fondo_pared, fondo_medio, fluo_neta, fluo_total, filas, toca_salida).

Usage:
    python fluorescencia_linaje.py ../October/16/Stacks/01-16102026-BF.tif --fluo GFP
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canales_enderezados as CE  # noqa: E402

COLUMNAS = 5          # central channel columns that are averaged (the cell sits in the centre)
COLUMNAS_PARED = 3    # columns at each edge of the crop (PDMS walls) used for the background
RECORTE_POLO = 2      # px removed at each pole (the blurry edge of the cell)
HOLGURA_VACIO = 3     # px around each cell that do not count as "empty channel"


def fluorescencia_linaje(stack_bf, fluo='GFP', canales=None):
    ruta = Path(stack_bf)
    base = ruta.stem.replace('-BF', '')
    canales = Path(canales) if canales else ruta.parent / 'Canales'
    geo = pd.read_csv(canales / f'{base}_canales_BF_geometria.csv')
    F = pd.read_csv(canales / f'{base}_canales_BF_linaje.csv')
    F = F[F.visible.astype(bool)]
    st = tifffile.imread(ruta.parent / f'{base}-{fluo}.tif').astype(np.float32)
    T = len(st)
    falta = {t for t in range(T) if not st[t].any()}     # black slices: no image for that fluorescence (SIN_<FLUO>)
    filas, vacio = [], []                               # vacio: (image, interior − wall) of cell-free stretches
    for _, g in geo.iterrows():
        if bool(g.get('al_borde', False)): continue
        R = CE.recortes(st, g, int(g.ancho), int(g.margen_sup), int(g.largo))   # (T, largo, ancho)
        W = R.shape[2]; c0 = W // 2 - COLUMNAS // 2; c1 = c0 + COLUMNAS
        centro = R[:, :, c0:c1].mean(2)                                    # (T, largo)
        pared = np.median(np.concatenate([R[:, :, :COLUMNAS_PARED], R[:, :, -COLUMNAS_PARED:]], 2), 2)
        Fc = F[F.canal == g.canal]
        fondo, salida = int(g.fila_fondo), int(g.fila_salida)
        for t in range(T):
            if t in falta: continue
            ft = Fc[Fc.foto == t + 1]
            ocup = np.zeros(R.shape[1], bool)
            for r in ft.itertuples():
                a = int(np.floor(r.fila_sup + 0.5)) + RECORTE_POLO
                b = int(np.ceil(r.fila_inf + 0.5)) - RECORTE_POLO
                b = min(b, salida - 1)
                ocup[max(int(r.fila_sup) - HOLGURA_VACIO, 0):int(r.fila_inf) + HOLGURA_VACIO + 1] = True
                if b - a < 3: continue
                m = float(centro[t, a:b].mean()); pw = float(np.median(pared[t, a:b]))
                filas.append((int(g.canal), t + 1, int(r.celula), m, pw, b - a, bool(r.sale_del_canal),
                              r.longitud_px if r.longitud_px == r.longitud_px else (r.fila_inf - r.fila_sup)))
            e = np.arange(fondo + 3, salida - 10)
            e = e[~ocup[e]]
            if len(e) >= 15:
                vacio.append((t + 1, float(centro[t, e].mean() - np.median(pared[t, e]))))
    M = pd.DataFrame(filas, columns=['canal', 'foto', 'celula', 'fluo_media', 'fondo_pared', 'filas', 'toca_salida', 'L'])
    V = pd.DataFrame(vacio, columns=['foto', 'medio'])
    medio = V.groupby('foto').medio.median().reindex(range(1, T + 1)).interpolate(limit_direction='both').fillna(0)
    M['fondo_medio'] = M.foto.map(medio)
    M['fluo_neta'] = M.fluo_media - M.fondo_pared - M.fondo_medio
    M['fluo_total'] = M.fluo_neta * M.L
    M = M.drop(columns='L').round(2)
    destino = canales / f'{base}_canales_BF_linaje_{fluo}.csv'
    M.to_csv(destino, index=False)
    print(f'{destino.name}: {len(M)} cell-images; medium background {medio.median():.1f}'
          + (f'; {len(falta)} images without {fluo} (not measured)' if falta else ''))
    return M


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    p = argparse.ArgumentParser(description='Fluorescence of each lineage cell.')
    p.add_argument('stacks', nargs='+', help='aligned bright-field stacks (-BF.tif)')
    p.add_argument('--fluo', default='GFP')
    a = p.parse_args()
    for s in a.stacks:
        fluorescencia_linaje(s, a.fluo)
