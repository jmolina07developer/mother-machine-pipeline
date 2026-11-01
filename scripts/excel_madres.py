"""
Excel sheet with the length of the mother cell (the one at the closed end of the channel, old pole) of
each channel in each image, to review and complete by hand.

One sheet per position: <Stacks>/Excel/<pos>-<fecha>_longitud_madres.xlsx

    row 1: time of each image (HH:MM:SS)
    row 2: image number in the stack
    then, two rows per channel:
      - the upper one: length (px) of the cell at the closed end of the channel in each image;
      - the lower one: only in the first image where the mother's division is seen, the length of the
        daughter (so mother + daughter in that image is the mother's length before dividing); empty elsewhere.
    column A: channel number (the same one drawn on the Lineage stacks).

Cells without a measurement (NaN) and their colour:
    red       the cell goes past the channel exit: measure by hand;
    yellow    the cell is there, but there is no reliable length (the network did not see it in that image
              and its box was interpolated, or the box does not match the expected length): review;
    grey      image flagged as defective;
    empty     there is no cell in the channel.
In light green, the lengths from images you corrected by hand (Cajas_Canales.ijm). If the top cell
changes without a division (the mother disappears or is lost), the cell carries a comment.

The cell at the closed end of the channel is, in each image, the cell of the automatic lineage closest to
the closed end of the channel. Channels cut by the image border are not measured and do not appear.

Usage:
    python excel_madres.py ../Ily/Stacks/*-BF.tif
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROJO, AMARILLO, GRIS, VERDE = 'FFFF7B7B', 'FFFFE699', 'FFD9D9D9', 'FFE2EFDA'
DECIMALES = 1


def _horas(stack_bf, base, canales, T):
    """Time (HH:MM:SS) of each image: from the stack log; otherwise from each slice label; otherwise, minutes."""
    reg = Path(stack_bf).with_name(f'{base}_registro.csv')
    if reg.exists():
        r = pd.read_csv(reg)
        if 'hora' in r and len(r) >= T:
            return [str(h).split(' ')[-1] for h in r.sort_values('slice').hora.iloc[:T]]
    try:
        import tifffile
        with tifffile.TiffFile(stack_bf) as tf:
            et = tf.imagej_metadata.get('Labels') if tf.imagej_metadata else None
        if et and len(et) >= T:
            hs = [re.search(r'_\d{6}_(\d{2})(\d{2})(\d{2})_', str(e)) for e in et[:T]]
            if all(hs): return [f'{m.group(1)}:{m.group(2)}:{m.group(3)}' for m in hs]
    except Exception:
        pass
    cp = canales / f'{base}_canales_BF_crecimiento_posicion.csv'
    if cp.exists():
        return [f'{m:.1f} min' for m in pd.read_csv(cp).tiempo_min.iloc[:T]]
    return [''] * T


def _malas(canales, base):
    zp = canales / f'{base}_canales_BF_cajas.zip'
    if not zp.exists(): return pd.DataFrame(columns=['canal', 'foto'])
    from leer_cajas import fotos_malas
    return fotos_malas(zp)


def madres_canal(F, C, fotos):
    """
    For one channel: {foto: dict(L, estado, nota)} for the mother and {foto: dict(L, estado, nota)} for the daughter.
    estado: 'ok', 'sale' (exits), 'sin_medida' (no measurement), 'manual' (ok, from your boxes).
    """
    madre_de = C.set_index('celula').madre.to_dict() if len(C) else {}
    hijas_de = C.groupby('madre').celula.apply(list).to_dict() if len(C) and C.madre.notna().any() else {}
    final_de = C.set_index('celula').final.to_dict() if len(C) else {}

    def celda(r):
        if bool(r.sale_del_canal): return dict(L=np.nan, estado='sale')
        if pd.isna(r.longitud_px): return dict(L=np.nan, estado='sin_medida')
        return dict(L=float(r.longitud_px), estado='manual' if bool(getattr(r, 'manual', False)) else 'ok')

    M, H = {}, {}
    prev = None                                              # top cell in the last image that had a cell
    V = F[F.visible.astype(bool)]
    for f in fotos:
        g = V[V.foto == f].sort_values('fila_sup')
        if not len(g): continue
        top = g.iloc[0]; cid = int(top.celula)
        M[f] = celda(top)
        if prev is not None and cid != prev:
            if madre_de.get(cid) == prev:                    # it has divided: the lower one is the daughter
                otras = [h for h in hijas_de.get(prev, []) if h != cid]
                gh = g[g.celula.isin(otras)]
                if len(gh):
                    cs = [celda(r) for r in gh.itertuples()]
                    if any(c['estado'] == 'sale' for c in cs): H[f] = dict(L=np.nan, estado='sale')
                    elif any(c['estado'] == 'sin_medida' for c in cs): H[f] = dict(L=np.nan, estado='sin_medida')
                    else:
                        H[f] = dict(L=sum(c['L'] for c in cs),
                                    estado='manual' if all(c['estado'] == 'manual' for c in cs) else 'ok')
                    if len(gh) > 1:
                        H[f]['nota'] = f'Fragmentación: suma de {len(gh)} hijas'
                else:
                    H[f] = dict(L=np.nan, estado='sin_medida', nota='División, pero la hija no se ve en esta foto')
            elif str(final_de.get(prev, '')).startswith('se pierde'):
                pass                                         # long gap of bad images: tracking starts over
            elif prev in set(g.celula.astype(int)):
                M[f]['nota'] = f'La célula de arriba cambia sin división (célula {prev} -> {cid}): aparece una célula por encima'
            else:
                M[f]['nota'] = (f'La célula de arriba cambia sin división (célula {prev} -> {cid}): '
                                f'la anterior {final_de.get(prev, "deja de verse")}')
        prev = cid
    return M, H


def excel_madres(stack_bf, salida=None, canales=None):
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    ruta = Path(stack_bf)
    base = ruta.stem.replace('-BF', '').replace('_canales_BF', '')
    canales = Path(canales) if canales else ruta.parent / 'Canales'
    salida = Path(salida) if salida else ruta.parent / 'Excel'
    salida.mkdir(parents=True, exist_ok=True)
    geo = pd.read_csv(canales / f'{base}_canales_BF_geometria.csv')
    F = pd.read_csv(canales / f'{base}_canales_BF_linaje.csv')
    C = pd.read_csv(canales / f'{base}_canales_BF_linaje_ciclos.csv')
    T = int(F.foto.max()) if len(F) else int(geo.fotos.iloc[0])
    fotos = list(range(1, T + 1))
    horas = _horas(ruta, base, canales, T)
    malas = _malas(canales, base)

    wb = Workbook(); ws = wb.active; ws.title = base[:31]
    fill = {k: PatternFill('solid', start_color=v, end_color=v) for k, v in
            dict(sale=ROJO, sin_medida=AMARILLO, mala=GRIS, manual=VERDE).items()}
    negrita = Font(bold=True); centro = Alignment(horizontal='center')
    ws.cell(1, 1, 'hora').font = negrita; ws.cell(2, 1, 'foto').font = negrita
    for j, f in enumerate(fotos):
        ws.cell(1, j + 2, horas[j]).alignment = centro
        ws.cell(2, j + 2, f).alignment = centro
        ws.cell(2, j + 2).font = negrita
    fila = 3; n_rojo = n_amar = 0
    for _, g in geo.sort_values('canal').iterrows():
        c = int(g.canal)
        if bool(g.get('al_borde', False)): continue
        mal = set(malas.foto[(malas.canal == c) | (malas.canal == 0)].astype(int)) if len(malas) else set()
        M, H = madres_canal(F[F.canal == c], C[C.canal == c], fotos)
        ws.cell(fila, 1, c).font = negrita
        for d, r in ((M, fila), (H, fila + 1)):
            for j, f in enumerate(fotos):
                cel = ws.cell(r, j + 2)
                if f in mal:
                    if d is M: cel.value = 'NaN'; cel.fill = fill['mala']
                    continue
                if f not in d: continue
                v = d[f]
                if v['estado'] in ('sale', 'sin_medida'):
                    cel.value = 'NaN'; cel.fill = fill[v['estado']]
                    n_rojo += v['estado'] == 'sale'; n_amar += v['estado'] == 'sin_medida'
                else:
                    cel.value = round(v['L'], DECIMALES)
                    if v['estado'] == 'manual': cel.fill = fill['manual']
                if v.get('nota'): cel.comment = Comment(v['nota'], 'linaje')
        fila += 2
    ws.freeze_panes = 'B3'
    ws.column_dimensions['A'].width = 7
    for j in range(len(fotos)): ws.column_dimensions[get_column_letter(j + 2)].width = 9
    # notes sheet
    wn = wb.create_sheet('Notas')
    notas = [
        ('Longitud de la célula madre (célula del final del canal, polo viejo) en píxeles.', None),
        ('Fila 1: hora de cada foto. Fila 2: número de foto en el stack. Columna A: número del canal (como en los stacks de Linaje).', None),
        ('Por canal, dos filas: arriba la madre en cada foto; abajo, solo en la primera foto en que se ve su división, la hija '
         '(madre + hija = longitud antes de dividirse).', None),
        ('NaN en rojo: la célula pasa la salida del canal (medir a mano).', 'sale'),
        ('NaN en amarillo: la célula está pero no hay longitud fiable (caja interpolada o que no cuadra): revisar.', 'sin_medida'),
        ('NaN en gris: foto marcada como defectuosa.', 'mala'),
        ('Verde: longitud de una foto corregida a mano (tus cajas).', 'manual'),
        ('Vacía: no hay ninguna célula en el canal en esa foto.', None),
        ('Comentario en una celda: la célula de arriba cambia sin división, o división en fragmentos.', None),
        (f'Generado a partir de {base}_canales_BF_linaje.csv (linaje automático).', None),
    ]
    for i, (t, k) in enumerate(notas):
        cel = wn.cell(i + 1, 1, t)
        if k: cel.fill = fill[k]
    wn.column_dimensions['A'].width = 120
    destino = salida / f'{base}_longitud_madres.xlsx'
    wb.save(destino)
    n_can = (fila - 3) // 2
    print(f'{destino.name}: {n_can} channels, {T} images; {n_rojo} cells in red (exit), {n_amar} in yellow')
    return destino


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    p = argparse.ArgumentParser(description='Excel sheet with the mother-cell length of each channel.')
    p.add_argument('stacks', nargs='+', help='aligned bright-field stacks (-BF.tif)')
    p.add_argument('--salida', help='output folder (default <Stacks>/Excel)')
    a = p.parse_args()
    for s in a.stacks:
        excel_madres(s, a.salida)
