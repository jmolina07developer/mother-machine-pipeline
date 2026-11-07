"""
Builds, for each position, two paired and aligned stacks from the individual
microscope images:

    <pos>-<ddmmaaaa>-BF.tif   bright field
    <pos>-<ddmmaaaa>-PI.tif   propidium iodide fluorescence (dead cells)

- All images have the same name (..._Regrowth_BF_...), so the channel is
  recognised from the image itself: bright field is much brighter than fluorescence.
- Each bright-field image is paired with the PI image closest in time
  (within MAX_SEGUNDOS). Slice n of one stack corresponds to slice n of the other.
  If a bright-field image has no PI, the PI slice is left black and labelled SIN_PI.
  If there are two bright-field images a few seconds apart (the PI image was taken in bright field),
  the first is used and the second is discarded.
- The alignment is computed from the chip numbers in the bright field and applied
  identically to both stacks.
- Slice names are those of the original files (with the time).
- Bright field is saved as 8-bit (half the size; more than enough to measure lengths),
  with the same scale in every slice. PI stays 16-bit to measure fluorescence.
- For each position it saves <pos>-<ddmmaaaa>_registro.csv: pairing,
  shifts and alignment quality.
- The original images are not modified.
- Experiments without PI (e.g. October/9, Ily 13/10): there is PI only if the images split into two
  groups of very different brightness (fluorescence less than half as bright as bright field)
  and some PI image is right next to a bright-field one. Otherwise all are bright field (even if the
  brightness changes with the medium) and only the BF stack is saved. This can be forced with --solo-bf.
- Two different fields with the same position number (the microscope images two sites on the
  chip, one after the other, and saves them with the same number; e.g. Ily 13/10, positions 06 and 11): if
  a position has two bright-field images taken almost at the same time and they are not the same field,
  they are split into two series, <pos> and <pos>b, following each field from one image to the next (the
  one that moves least). If they are the same field (repeated image), the first is used as before.

- Two bright-field images of the same position a few seconds apart: the one kept is the one that has
  fluorescence and best matches its area (quitar_bf_repetidos); previously the first was always kept.
- Areas with shifted numbering (in some round you lose track and save area 11 as 10, 12 as 11...):
  each bright-field image is compared (chip numbers) with a reference image of its own position and of
  the two neighbouring ones; if it matches a neighbour much better (correlation >= AREA_ENCAJE and AREA_VENTAJA
  higher than with its own), it is moved to that position, together with its fluorescence image. The changes are
  saved in <Stacks>/areas_reasignadas.csv and the correlations in <Stacks>/areas_comprobacion.csv (they are reused).
- The fluorescence can be PI or GFP (or another): `fluo` only changes the stack name (<pos>-<fecha>-GFP.tif).
- Two fluorescences in the same experiment (e.g. October/22: GFP in every round and PI in a few):
  `--fluo GFP PI`. Fluorescence images are split by brightness into as many groups as names
  (Otsu on the log of the median), from brightest to dimmest in the given order. Each bright-field
  image is paired with the closest image of each fluorescence and one stack per fluorescence is saved,
  all with the same slices as the bright-field one (black and labelled SIN_<FLUO> where missing).
- Empty images (0 bytes, e.g. a partially copied file) are ignored with a warning.

Usage from the terminal:
    python generar_stacks.py "ruta/a/October/2/Images"
    python generar_stacks.py "ruta/a/October/2/Images" --posiciones 01 02 03
    python generar_stacks.py "ruta/a/October/2/Images" --salida "ruta/a/Stacks" --sin-alinear
    python generar_stacks.py "ruta/a/October/9/Images" --solo-bf
    python generar_stacks.py "ruta/a/October/16/Images" --fluo GFP
    python generar_stacks.py "ruta/a/October/22/Images" --fluo GFP PI

Requires: pip install tifffile scikit-image pandas
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alinear_stack import (ZONA_NUMEROS, aplicar_desplazamientos,  # noqa: E402
                           avisos, calcular_desplazamientos, comprobar_alineacion,
                           medir_giros, aplicar_giros, GIRO_MIN)

PATRON = re.compile(r'^(\d+)_(\d{6})_(\d{6})_.*\.tiff?$')
MAX_SEGUNDOS = 60      # maximum time gap between a bright-field image and its PI image
MARGEN_STACKS = 80     # maximum shift searched (px); this experiment has jumps of ~45 px
POS_MIN_IMAGENES = 10  # positions with fewer images are ignored (tests)
FRACCION_PI = 0.5      # fluorescence has less than this fraction of the bright-field brightness
MISMO_CAMPO = 25       # px: two near-simultaneous images shifted by less than this are the same field
AREA_ENCAJE = 0.85     # an image that matches the reference of a neighbouring position this well...
AREA_VENTAJA = 0.15    # ...and this much better than its own position's belongs to the neighbour (shifted area number)
AREA_SALTO_MAX = 6     # if it matches neither its own position nor the neighbours, jumps of up to ±6 areas are tried
                       # (KAN-TRIM 19/04/2024: one round saved with the number shifted by 4 areas). The whole chip is
                       # not searched: the numbers repeat in each lane (October/22: 05 and 15 show 0100–0115)


def inventario(carpeta, fluos=('PI',)):
    """One row per image: position, time, brightness (median) and channel (BF / PI). With several fluorescences
    (fluos = ('GFP', 'PI')), the fluorescence images are split among them by brightness (from brightest to
    dimmest, in the given order)."""
    filas, vacias = [], []
    for f in sorted(Path(carpeta).iterdir()):
        m = PATRON.match(f.name)
        if not m:
            continue
        if f.stat().st_size == 0:
            vacias.append(f.name); continue
        img = tifffile.memmap(f)
        filas.append(dict(pos=m.group(1),
                          hora=pd.to_datetime(m.group(2) + m.group(3), format='%y%m%d%H%M%S'),
                          mediana=float(np.median(img[::10, ::10])), archivo=f.name))
    if vacias:
        print(f'WARNING: {len(vacias)} empty images (0 bytes) are ignored: {vacias[:5]}{" …" if len(vacias) > 5 else ""}')
    inv = pd.DataFrame(filas)
    if inv.empty:
        raise FileNotFoundError(f'No images matching the expected pattern in {carpeta}')
    # Threshold halfway between the typical brightness of each group (Otsu is only used to separate them)
    m = inv['mediana'].to_numpy()
    t = threshold_otsu(m) if len(np.unique(m)) > 1 else m[0]
    bajo, alto = (np.median(m[m <= t]), np.median(m[m > t])) if (m > t).any() and (m <= t).any() else (m.max(), m.max())
    umbral = (bajo + alto) / 2
    if bajo < FRACCION_PI * alto:
        inv['canal'] = np.where(inv['mediana'] > umbral, 'BF', 'PI')
    else:                       # a single brightness group: everything is bright field
        inv['canal'] = 'BF'; umbral = float('nan')
    fl = inv['canal'] == 'PI'
    if len(fluos) > 1 and fl.sum() >= len(fluos):
        from skimage.filters import threshold_multiotsu
        lm = np.log(inv.loc[fl, 'mediana'].to_numpy())
        cortes = np.sort(threshold_multiotsu(lm, classes=len(fluos)))[::-1]    # from brightest to dimmest
        clase = np.full(len(lm), len(fluos) - 1)
        for k in range(len(cortes) - 1, -1, -1):
            clase[lm > cortes[k]] = k
        inv.loc[fl, 'canal'] = [fluos[k] for k in clase]
        print('Fluorescences separated by brightness (image median): ' + ', '.join(
            f'{f}: {(inv.canal == f).sum()} images, median {inv.mediana[inv.canal == f].median():.0f}' for f in fluos)
              + ' | cutoffs: ' + ', '.join(f'{np.exp(c):.0f}' for c in cortes))
    elif len(fluos) == 1:
        inv.loc[fl, 'canal'] = 'PI'
    return inv.sort_values(['pos', 'hora']).reset_index(drop=True), umbral


def hay_pi(inv, max_seg=MAX_SEGUNDOS):
    """Are there PI images? Only if there are images from both brightness groups and some PI image is less than
    max_seg away from a bright-field image of the same position."""
    if (inv['canal'] != 'BF').sum() == 0:
        return False
    for _, g in inv.groupby('pos'):
        g = g.sort_values('hora')
        dt = g['hora'].diff().dt.total_seconds().to_numpy(); c = g['canal'].to_numpy()
        if any(dt[i] < max_seg and c[i] != c[i - 1] for i in range(1, len(g))):
            return True
    return False


def _desplazamiento(a, b, margen):
    """(dy, dx, quality) that maps image b onto image a (edges, already downscaled)."""
    from skimage.feature import match_template
    plantilla = b[margen:-margen, margen:-margen]
    ncc = match_template(a, plantilla)
    r, c = np.unravel_index(np.argmax(ncc), ncc.shape)
    return margen - r, margen - c, float(ncc[r, c])


def separar_series(carpeta, inv_pos, max_seg=MAX_SEGUNDOS, zona=ZONA_NUMEROS, reducir=2, margen=50):
    """
    If a position (bright field only) has two near-simultaneous images that are NOT the same field, they are
    two sites on the chip saved with the same number: they are split into two series by following each field
    (each image goes to the series whose last image matches it best: correlation of the chip numbers,
    searching at most `margen` px; the numbers repeat along the chip, so a large margin
    would confuse the two fields). Returns a list of inventories
    (one per series; the second with pos = <pos>b). If they are the same field, returns [inv_pos].
    """
    from alinear_stack import _bordes
    inv_pos = inv_pos.sort_values('hora').reset_index(drop=True)
    if (inv_pos['canal'] != 'BF').any():
        return [inv_pos]
    dt = inv_pos['hora'].diff().dt.total_seconds().fillna(np.inf).to_numpy()
    grupos, cur = [], []
    for i in range(len(inv_pos)):
        if cur and dt[i] >= max_seg:
            grupos.append(cur); cur = []
        cur.append(i)
    grupos.append(cur)
    dobles = [g for g in grupos if len(g) >= 2]
    if len(dobles) < 0.3 * len(grupos):
        return [inv_pos]                                    # an isolated repeated image: as before
    y0, y1 = zona; m = margen // reducir
    cache = {}
    def B(i):
        if i not in cache:
            img = tifffile.imread(Path(carpeta) / inv_pos.archivo[i])
            cache[i] = _bordes(img)[y0:y1:reducir, ::reducir].astype(np.float32)
        return cache[i]
    def enc(i, j):
        dy, dx, q = _desplazamiento(B(i), B(j), m)
        return q, float(np.hypot(dy, dx) * reducir)
    def dist(i, j):                                         # lower = better match
        return -enc(i, j)[0]
    # same field? (the two near-simultaneous images match very well with almost no shift)
    pares = [enc(g[0], g[1]) for g in dobles[:5]]
    q_par = float(np.median([q for q, _ in pares])); d_par = float(np.median([d for _, d in pares]))
    if q_par > 0.85 and d_par < MISMO_CAMPO:
        return [inv_pos]
    serie = np.full(len(inv_pos), -1)
    g0 = dobles[0]; serie[g0[0]] = 0; serie[g0[1]] = 1
    ultimo = {0: g0[0], 1: g0[1]}
    for g in grupos:
        if all(serie[i] >= 0 for i in g): continue
        libres = [i for i in g if serie[i] < 0]
        if len(libres) == 1:
            i = libres[0]; s_ = min((0, 1), key=lambda s: dist(ultimo[s], i))
            serie[i] = s_; ultimo[s_] = i
        else:
            a, b = libres[0], libres[1]
            c01 = dist(ultimo[0], a) + dist(ultimo[1], b); c10 = dist(ultimo[0], b) + dist(ultimo[1], a)
            sa, sb = (0, 1) if c01 <= c10 else (1, 0)
            serie[a] = sa; serie[b] = sb; ultimo[sa] = a; ultimo[sb] = b
            for i in libres[2:]:
                s_ = min((0, 1), key=lambda s: dist(ultimo[s], i)); serie[i] = s_; ultimo[s_] = i
    pos = inv_pos['pos'].iloc[0]
    out = []
    for s_, suf in ((0, ''), (1, 'b')):
        sub = inv_pos[serie == s_].copy(); sub['pos'] = pos + suf
        out.append(sub.reset_index(drop=True))
    print(f'Position {pos}: two different fields with the same number (the near-simultaneous images match {q_par:.2f}): '
          f'split into {pos} ({len(out[0])} images) and {pos}b ({len(out[1])} images)')
    return out


def corregir_areas(carpeta, inv, salida=None, max_seg=MAX_SEGUNDOS, zona=ZONA_NUMEROS, reducir=2, margen=80,
                   encaje=AREA_ENCAJE, ventaja=AREA_VENTAJA, tiempo_max=None):
    """
    Detects images saved with a shifted area number (e.g. a round in which area 11 was saved
    as 10, area 12 as 11...) by comparing the chip numbers of each bright-field image with a
    reference image (the one in the middle of the movie) of its own position and of the two neighbouring ones. If it
    matches a neighbour much better, the image (and its fluorescence, the closest one with the same number) moves to that position.
    Returns (corrected inventory, table of changes).
    tiempo_max (s): if given, saves what has been computed so far to areas_comprobacion.csv and raises TimeoutError; when
    called again it resumes where it left off (for computers where a process cannot run for more than a few minutes).
    """
    import time as _time
    t_ini = _time.time()
    from alinear_stack import _bordes
    carpeta = Path(carpeta)
    inv = inv.copy(); inv['pos_original'] = inv['pos']
    cuenta = inv[inv.canal == 'BF'].groupby('pos').size()
    posiciones = sorted(p for p in cuenta.index if cuenta[p] >= POS_MIN_IMAGENES)
    if len(posiciones) < 2:
        return inv, pd.DataFrame()
    ancho = len(posiciones[0])
    y0, y1 = zona; m = margen // reducir
    def B(f):
        return _bordes(tifffile.imread(carpeta / f))[y0:y1:reducir, ::reducir].astype(np.float32)
    bf = inv[inv.canal == 'BF'].sort_values('hora')          # also the images of positions with few images
    refs = {p: B(g.archivo.iloc[len(g) // 2]) for p, g in bf[bf.pos.isin(posiciones)].groupby('pos')}
    nums = {int(p): p for p in posiciones}
    cache_f = Path(salida) / 'areas_comprobacion.csv' if salida else None
    previo = pd.read_csv(cache_f, dtype={'pos': str}) if cache_f and cache_f.exists() else pd.DataFrame()
    hecho = {} if previo.empty else {r.archivo: r for r in previo.itertuples()}
    filas = []
    def _guardar_parcial():
        hechos_ya = {f['archivo'] for f in filas}
        resto = previo[~previo.archivo.isin(hechos_ya)] if len(previo) else pd.DataFrame()
        pd.concat([pd.DataFrame(filas), resto], ignore_index=True).to_csv(cache_f, index=False)
    for r in bf.itertuples():
        if tiempo_max and cache_f and _time.time() - t_ini > tiempo_max:
            Path(salida).mkdir(parents=True, exist_ok=True); _guardar_parcial()
            raise TimeoutError(f'corregir_areas: {len(filas)} of {len(bf)} images checked; run again to continue')
        n0 = int(r.pos); b = None
        h = hecho.get(r.archivo)
        if h is not None and str(getattr(h, 'pos', '')) == r.pos:
            q = {-1: h.q_ant, 0: h.q_propia, 1: h.q_sig}
            salto, q_otra = getattr(h, 'salto_otra', np.nan), getattr(h, 'q_otra', np.nan)
            otra_hecha = ('busqueda' in previo.columns and getattr(h, 'busqueda', '') == f'salto{AREA_SALTO_MAX}'
                          and not (pd.isna(salto) and np.nanmax([q[-1], q[0], q[1], -1]) < encaje))
        else:
            b = B(r.archivo); q = {}
            for k in (-1, 0, 1):
                q[k] = _desplazamiento(b, refs[nums[n0 + k]], m)[2] if (n0 + k) in nums else np.nan
            salto, q_otra, otra_hecha = np.nan, np.nan, False
        # matches neither its own position nor the neighbours: larger jumps are tried
        if not otra_hecha and np.nanmax([q[-1], q[0], q[1], -1]) < encaje:
            if b is None: b = B(r.archivo)
            mejor = (np.nan, np.nan)
            for k in [s_ * d for s_ in range(2, AREA_SALTO_MAX + 1) for d in (-1, 1)]:
                if (n0 + k) in nums:
                    qk = _desplazamiento(b, refs[nums[n0 + k]], m)[2]
                    if not qk <= mejor[1]: mejor = (k, qk)
            salto, q_otra = mejor
        filas.append(dict(archivo=r.archivo, pos=r.pos, q_ant=q[-1], q_propia=q[0], q_sig=q[1], salto_otra=salto, q_otra=q_otra,
                          busqueda=f'salto{AREA_SALTO_MAX}'))
    Q = pd.DataFrame(filas)
    if cache_f:
        Path(salida).mkdir(parents=True, exist_ok=True); Q.to_csv(cache_f, index=False)
    cambios = []
    for r in Q.itertuples():
        qp = r.q_propia if r.q_propia == r.q_propia else 0.0     # position without a reference (few images): 0
        opciones = [(-1, r.q_ant), (1, r.q_sig)] + ([(int(r.salto_otra), r.q_otra)] if r.salto_otra == r.salto_otra else [])
        for k, qk in sorted(opciones, key=lambda o: -(o[1] if o[1] == o[1] else -1))[:1]:
            if qk == qk and qk >= encaje and qk >= qp + ventaja:
                nueva = f'{int(r.pos) + k:0{ancho}d}'
                cambios.append(dict(archivo=r.archivo, de=r.pos, a=nueva, encaje_propia=round(r.q_propia, 3),
                                    encaje_nueva=round(qk, 3)))
    qmap = dict(zip(Q.archivo, Q.q_propia))
    inv['q_area'] = inv.archivo.map(qmap)                # match of each bright-field image with its area (final)
    idx = inv.set_index('archivo').index
    for c in list(cambios):
        fila = inv.index[idx == c['archivo']][0]; h = inv.at[fila, 'hora']
        inv.at[fila, 'pos'] = c['a']; inv.at[fila, 'q_area'] = c['encaje_nueva']
        fl = inv[(inv.canal != 'BF') & (inv.pos_original == c['de'])]   # its fluorescence: the closest one
        if len(fl):
            dt = (fl.hora - h).dt.total_seconds().abs()
            if dt.min() <= max_seg:
                j = dt.idxmin(); inv.at[j, 'pos'] = c['a']
                cambios.append(dict(archivo=inv.at[j, 'archivo'], de=c['de'], a=c['a'], encaje_propia=np.nan, encaje_nueva=np.nan))
    C = pd.DataFrame(cambios)
    if len(C):
        nbf = int(C.encaje_nueva.notna().sum())
        print(f'Areas with shifted numbering: {nbf} bright-field images (and their fluorescence) moved to another position: '
              + ', '.join(f"{a[:2]}→{b}" for a, b in zip(C.archivo[C.encaje_nueva.notna()], C.a[C.encaje_nueva.notna()])))
        if salida: C.to_csv(Path(salida) / 'areas_reasignadas.csv', index=False)
    return inv.sort_values(['pos', 'hora']).reset_index(drop=True), C


def quitar_bf_repetidos(inv_pos, max_seg=MAX_SEGUNDOS):
    """
    Two (or more) bright-field images of the same position less than max_seg apart: only one is the image of that round.
    The one kept is the one with fluorescence less than max_seg away and, among those, the one that best matches the chip
    numbers of its area (q_area, from corregir_areas); on a tie, the first one. This keeps out both an image of another
    field taken just before (October/22, area 21: images of another site on the chip seconds before the good one) and an
    image that should have been a fluorescence image and came out in bright field. Returns (inventory without the extra images, extra images).
    """
    inv_pos = inv_pos.sort_values('hora').reset_index(drop=True)
    bf = inv_pos[inv_pos.canal == 'BF']; fl = inv_pos[inv_pos.canal != 'BF']
    grupos, cur, ultimo = [], [], None
    for k, h in bf.hora.items():
        if cur and (h - ultimo).total_seconds() >= max_seg:
            grupos.append(cur); cur = []
        cur.append(k); ultimo = h
    if cur: grupos.append(cur)
    fuera = []
    for g in grupos:
        if len(g) < 2: continue
        horas = inv_pos.loc[g, 'hora']
        duenos = set()                                      # each nearby fluorescence image belongs to its closest bright-field image
        for h_fl in fl.hora:
            dt = (horas - h_fl).dt.total_seconds().abs()
            if dt.min() <= max_seg: duenos.add(dt.idxmin())
        def clave(k):
            q = inv_pos.at[k, 'q_area'] if 'q_area' in inv_pos and inv_pos.at[k, 'q_area'] == inv_pos.at[k, 'q_area'] else -1.0
            return (k in duenos, round(q, 2), -g.index(k))
        mejor = max(g, key=clave)
        fuera += [k for k in g if k != mejor]
    return inv_pos.drop(index=fuera).reset_index(drop=True), inv_pos.archivo[fuera].tolist()


def emparejar(inv_pos, max_seg=MAX_SEGUNDOS, fluos=('PI',)):
    """Pairs each BF with the closest-in-time, not yet used image of each fluorescence.
    With a single fluorescence the columns are called archivo_PI / dt_BF_PI_s (as always); with several,
    archivo_<FLUO> / dt_BF_<FLUO>_s."""
    inv_pos, quitadas = quitar_bf_repetidos(inv_pos, max_seg)
    if len(fluos) > 1:
        P = None; sobr_tot = []; rep_tot = quitadas
        for f in fluos:
            sub = inv_pos[inv_pos.canal.isin(['BF', f])].copy(); sub.loc[sub.canal == f, 'canal'] = 'PI'
            Pf, sobr, rep_ = _emparejar_uno(sub, max_seg, descartar_extra=False)
            Pf = Pf.rename(columns={'archivo_PI': f'archivo_{f}', 'dt_BF_PI_s': f'dt_BF_{f}_s'})
            P = Pf if P is None else P.merge(Pf[['archivo_BF', f'archivo_{f}', f'dt_BF_{f}_s']], on='archivo_BF')
            sobr_tot += sobr
        con_fluo = np.column_stack([P[f'archivo_{f}'] != '' for f in fluos]).any(1)
        P, extra = _descartar_extra(P, con_fluo)
        return P, sobr_tot, rep_tot + extra
    P, sobr, rep_ = _emparejar_uno(inv_pos, max_seg)
    return P, sobr, quitadas + rep_


def _descartar_extra(P, con_fluo):
    """Extra bright-field image, without fluorescence, when the previous one is paired (in stretches without
    fluorescence, e.g. after a medium change, nothing is discarded): (a) less than half an interval after the
    previous one, or (b) between two paired images and not matching the chip numbers of its area (q_area < 0.5:
    an image of another site on the chip; October/22: one from area 01 saved as 30 between two rounds)."""
    if not con_fluo.any() or len(P) <= 3:
        return P, []
    dt = P.hora.diff().dt.total_seconds()
    anterior = np.r_[False, con_fluo[:-1]]; siguiente = np.r_[con_fluo[1:], False]
    q = P['q_area'].to_numpy(float) if 'q_area' in P else np.full(len(P), np.nan)
    extra = (~con_fluo) & anterior & ((dt < 0.5 * dt.median()).to_numpy() | (siguiente & (q < 0.5)))
    return P[~extra].reset_index(drop=True), P.archivo_BF[extra].tolist()


def _emparejar_uno(inv_pos, max_seg=MAX_SEGUNDOS, descartar_extra=True):
    bf = inv_pos[inv_pos.canal == 'BF'].reset_index(drop=True)
    pi = inv_pos[inv_pos.canal == 'PI'].reset_index(drop=True)
    # Two BF images a few seconds apart: the second one should have been the PI image and came out in bright field.
    # The first one is kept and the second one is discarded.
    repetidos, ultimo = [], None
    for k, h in bf['hora'].items():
        if ultimo is not None and (h - ultimo).total_seconds() < max_seg:
            repetidos.append(k)
        else:
            ultimo = h
    bf_repetidos = bf.loc[repetidos, 'archivo'].tolist()
    bf = bf.drop(index=repetidos).reset_index(drop=True)
    usados, filas = set(), []
    for _, b in bf.iterrows():
        dt = (pi['hora'] - b['hora']).dt.total_seconds().abs()
        dt[list(usados)] = np.inf
        j = dt.idxmin() if len(dt) else None
        if j is not None and dt[j] <= max_seg:
            usados.add(j)
            filas.append(dict(hora=b.hora, archivo_BF=b.archivo, archivo_PI=pi.archivo[j],
                              dt_BF_PI_s=(pi.hora[j] - b.hora).total_seconds(), q_area=b.get('q_area', np.nan)))
        else:
            filas.append(dict(hora=b.hora, archivo_BF=b.archivo, archivo_PI='', dt_BF_PI_s=np.nan, q_area=b.get('q_area', np.nan)))
    sobrantes = pi.drop(index=list(usados))['archivo'].tolist()
    P = pd.DataFrame(filas)
    if len(pi) and descartar_extra:
        P, extra = _descartar_extra(P, (P.archivo_PI != '').to_numpy())
        bf_repetidos += extra
    return P, sobrantes, bf_repetidos


def igualar_exposicion(stack, tolerancia=0.10):
    """
    If the bright-field brightness changes between images (October/22: the exposure was changed at 18:07 and some
    rounds came out darker), each image is mapped linearly onto the reference: its 1st percentile and its median
    (the background) become the median of those of all images. Only if the difference between images exceeds
    `tolerancia`; otherwise the stack is left untouched. Returns (stack, number of corrected images).
    """
    sub = stack[:, ::4, ::4].reshape(len(stack), -1).astype(np.float32)
    p1, med = np.percentile(sub, 1, axis=1), np.median(sub, axis=1)
    rango = med - p1
    r_p1, r_rango = np.median(p1), np.median(rango)
    cambia = np.abs(rango / r_rango - 1) > tolerancia
    if not cambia.any():
        return stack, 0
    out = stack.astype(np.float32)
    for i in np.where(cambia)[0]:
        out[i] = r_p1 + (out[i] - p1[i]) * (r_rango / rango[i])
    info = np.iinfo(stack.dtype) if np.issubdtype(stack.dtype, np.integer) else None
    if info is not None:
        out = np.clip(np.round(out), info.min, info.max).astype(stack.dtype)
    return out, int(cambia.sum())


def a_8bits(stack, percentiles=(0.1, 99.9)):
    """Converts a stack to 8 bits with the SAME scale for all slices (no flicker). First it equalises the
    exposure of the images if it changes (igualar_exposicion)."""
    stack, n = igualar_exposicion(stack)
    if n:
        print(f'   exposure equalised in {n} bright-field images', flush=True)
    lo, hi = np.percentile(stack[:, ::4, ::4], percentiles)
    out = np.clip((stack.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
    return np.round(out).astype(np.uint8), float(lo), float(hi)


def generar_posicion(carpeta, inv_pos, salida, alinear=True, zona=ZONA_NUMEROS, margen=MARGEN_STACKS,
                     bf_8bits=True, solo_bf=False, fluo='PI'):
    fluos = [fluo] if isinstance(fluo, str) else list(fluo)
    if len(fluos) > 1 and not solo_bf:
        return _generar_posicion_varias(carpeta, inv_pos, salida, alinear, zona, margen, bf_8bits, fluos)
    carpeta, salida = Path(carpeta), Path(salida)
    pos = inv_pos['pos'].iloc[0]
    fecha = inv_pos['hora'].iloc[0].strftime('%d%m%Y')
    base = f'{pos}-{fecha}'

    pares, sobrantes, bf_repetidos = emparejar(inv_pos)
    bf = np.stack([tifffile.imread(carpeta / f) for f in pares.archivo_BF])
    pi = np.zeros_like(bf)
    for i, f in enumerate(pares.archivo_PI):
        if f:
            pi[i] = tifffile.imread(carpeta / f)
    etiquetas_bf = list(pares.archivo_BF)
    etiquetas_pi = [f if f else f'SIN_PI ({b})' for f, b in zip(pares.archivo_PI, pares.archivo_BF)]

    registro = pares.copy()
    registro.insert(0, 'slice', np.arange(1, len(pares) + 1))
    if alinear:
        giros = medir_giros(bf)                                  # rotated chip: straightened before shifting
        bf = aplicar_giros(bf, giros); pi = aplicar_giros(pi, giros)
        registro['giro_grados'] = giros
        desp = calcular_desplazamientos(bf, zona, margen)
        bf = aplicar_desplazamientos(bf, desp)
        pi = aplicar_desplazamientos(pi, desp)
        cols = ['dy_px', 'dx_px', 'calidad', 'comparado_con', 'salto_px']
        registro[cols] = desp[cols]
        avisos(desp, margen, nombre=f'Position {pos}:')
        # check: residual shift of each slice relative to the previous, already aligned one (0–1 px if all is well)
        res, enc = comprobar_alineacion(bf)
        registro['residuo_px'] = res; registro['encaje_anterior'] = np.round(enc, 3)
        dudosos = registro.slice[(registro.residuo_px > 2) | (registro.encaje_anterior < 0.8)].tolist()
    if bf_8bits:   # after aligning (the alignment is computed on the original 16-bit images)
        bf, escala_min, escala_max = a_8bits(bf)

    salida.mkdir(parents=True, exist_ok=True)
    guardar = [('BF', bf, etiquetas_bf)] + ([] if solo_bf else [(fluo, pi, etiquetas_pi)])
    if solo_bf:
        registro = registro.drop(columns=['archivo_PI', 'dt_BF_PI_s'])
    for sufijo, datos, etiquetas in guardar:
        tifffile.imwrite(salida / f'{base}-{sufijo}.tif', datos, imagej=True,
                         metadata={'Labels': etiquetas, 'axes': 'ZYX'})
    registro.to_csv(salida / f'{base}_registro.csv', index=False)

    sin_pi = 0 if solo_bf else int((registro.archivo_PI == '').sum())
    resumen = dict(pos=pos, slices=len(registro), sin_PI=sin_pi, PI_sobrantes=len(sobrantes),
                   BF_repetidos=len(bf_repetidos),
                   dt_BF_PI_max_s=np.nan if solo_bf else registro.dt_BF_PI_s.max())
    if bf_8bits:   # which 16-bit values correspond to 0 and 255 in the 8-bit BF stack
        resumen.update(BF_8bits_0=escala_min, BF_8bits_255=escala_max)
    if alinear:
        resumen.update(slice_referencia=desp.attrs['slice_referencia'],
                       slices_revisar=' '.join(map(str, sorted(set(registro.slice[registro.calidad < 0.8]) | set(dudosos)))),
                       residuo_max_px=float(registro.residuo_px.max()), giro_max=float(registro.giro_grados.abs().max()),
                       dx_min=registro.dx_px.min(), dx_max=registro.dx_px.max(),
                       dy_min=registro.dy_px.min(), dy_max=registro.dy_px.max(),
                       calidad_min=round(registro.calidad.min(), 3))
    msg = f'{base}: {len(registro)} slices'
    if sin_pi:
        msg += f' | {sin_pi} without PI (slices {registro.slice[registro.archivo_PI == ""].tolist()})'
    if sobrantes:
        msg += f' | unpaired PI images discarded: {sobrantes}'
    if bf_repetidos:
        msg += f' | repeated BF images discarded: {bf_repetidos}'
    if alinear:
        msg += (f' | reference: slice {resumen["slice_referencia"]}'
                f' | dx {resumen["dx_min"]}..{resumen["dx_max"]}, dy {resumen["dy_min"]}..{resumen["dy_max"]} px,'
                f' min quality {resumen["calidad_min"]} | check: max residual {resumen["residuo_max_px"]:.0f} px')
        if dudosos:
            msg += f' | DOUBTFUL ALIGNMENT in slices {dudosos} (check them)'
    print(msg, flush=True)
    return resumen


def _generar_posicion_varias(carpeta, inv_pos, salida, alinear, zona, margen, bf_8bits, fluos):
    """Like generar_posicion, with one stack per fluorescence (same slices as the bright field)."""
    carpeta, salida = Path(carpeta), Path(salida)
    pos = inv_pos['pos'].iloc[0]
    base = f'{pos}-{inv_pos["hora"].iloc[0].strftime("%d%m%Y")}'
    pares, sobrantes, bf_repetidos = emparejar(inv_pos, fluos=fluos)
    bf = np.stack([tifffile.imread(carpeta / f) for f in pares.archivo_BF])
    fl, etiquetas = {}, {'BF': list(pares.archivo_BF)}
    for f in fluos:
        st = np.zeros_like(bf)
        for i, a in enumerate(pares[f'archivo_{f}']):
            if a: st[i] = tifffile.imread(carpeta / a)
        fl[f] = st
        etiquetas[f] = [a if a else f'SIN_{f} ({b})' for a, b in zip(pares[f'archivo_{f}'], pares.archivo_BF)]
    registro = pares.copy(); registro.insert(0, 'slice', np.arange(1, len(pares) + 1))
    dudosos = []
    if alinear:
        giros = medir_giros(bf)
        bf = aplicar_giros(bf, giros)
        for f in fluos: fl[f] = aplicar_giros(fl[f], giros)
        registro['giro_grados'] = giros
        desp = calcular_desplazamientos(bf, zona, margen)
        bf = aplicar_desplazamientos(bf, desp)
        for f in fluos: fl[f] = aplicar_desplazamientos(fl[f], desp)
        cols = ['dy_px', 'dx_px', 'calidad', 'comparado_con', 'salto_px']
        registro[cols] = desp[cols]
        avisos(desp, margen, nombre=f'Position {pos}:')
        res, enc = comprobar_alineacion(bf)
        registro['residuo_px'] = res; registro['encaje_anterior'] = np.round(enc, 3)
        dudosos = registro.slice[(registro.residuo_px > 2) | (registro.encaje_anterior < 0.8)].tolist()
    if bf_8bits:
        bf, escala_min, escala_max = a_8bits(bf)
    salida.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(salida / f'{base}-BF.tif', bf, imagej=True, metadata={'Labels': etiquetas['BF'], 'axes': 'ZYX'})
    for f in fluos:
        tifffile.imwrite(salida / f'{base}-{f}.tif', fl[f], imagej=True, metadata={'Labels': etiquetas[f], 'axes': 'ZYX'})
    registro.to_csv(salida / f'{base}_registro.csv', index=False)
    resumen = dict(pos=pos, slices=len(registro), sobrantes=len(sobrantes), BF_repetidos=len(bf_repetidos))
    for f in fluos:
        resumen[f'con_{f}'] = int((registro[f'archivo_{f}'] != '').sum())
        resumen[f'dt_BF_{f}_max_s'] = registro[f'dt_BF_{f}_s'].max()
    if bf_8bits:
        resumen.update(BF_8bits_0=escala_min, BF_8bits_255=escala_max)
    if alinear:
        resumen.update(slice_referencia=desp.attrs['slice_referencia'],
                       slices_revisar=' '.join(map(str, sorted(set(registro.slice[registro.calidad < 0.8]) | set(dudosos)))),
                       residuo_max_px=float(registro.residuo_px.max()), giro_max=float(registro.giro_grados.abs().max()), dx_min=registro.dx_px.min(), dx_max=registro.dx_px.max(),
                       dy_min=registro.dy_px.min(), dy_max=registro.dy_px.max(), calidad_min=round(registro.calidad.min(), 3))
    msg = f'{base}: {len(registro)} slices | ' + ', '.join(f'{f} in {resumen[f"con_{f}"]}' for f in fluos)
    if sobrantes: msg += f' | unpaired fluorescence images discarded: {sobrantes}'
    if bf_repetidos: msg += f' | repeated BF images discarded: {bf_repetidos}'
    if alinear:
        msg += (f' | max rotation {resumen["giro_max"]:.2f}° | dx {resumen["dx_min"]}..{resumen["dx_max"]}, dy {resumen["dy_min"]}..{resumen["dy_max"]} px,'
                f' min quality {resumen["calidad_min"]} | max residual {resumen["residuo_max_px"]:.0f} px')
        if dudosos: msg += f' | DOUBTFUL ALIGNMENT in slices {dudosos} (check them)'
    print(msg, flush=True)
    return resumen


def quitar_descartadas(inv, salida):
    """
    Removes from the inventory the images listed in <salida>/imagenes_descartadas.csv (column `archivo`; `motivo`
    optional), written by hand: e.g. images of another field of the chip saved under a position (Ily 03/11: position
    08 at 16:03 and 16:31, position 20 at 14:40), which do not match the numbers of their position.
    """
    f = Path(salida) / 'imagenes_descartadas.csv'
    if not f.exists():
        return inv
    fuera = set(pd.read_csv(f).archivo.astype(str))
    n = inv.archivo.isin(fuera).sum()
    if n:
        print(f'{n} images discarded by hand ({f.name})')
    return inv[~inv.archivo.isin(fuera)].reset_index(drop=True)


def generar_stacks(carpeta, salida=None, posiciones=None, alinear=True,
                   zona=ZONA_NUMEROS, margen=MARGEN_STACKS, bf_8bits=True, solo_bf=None, fluo='PI', corregir=True):
    carpeta = Path(carpeta)
    salida = Path(salida) if salida else carpeta.parent / 'Stacks'
    fluos = [fluo] if isinstance(fluo, str) else list(fluo)
    inv, umbral = inventario(carpeta, fluos)
    inv = quitar_descartadas(inv, salida)
    if corregir:                                         # images saved with a shifted area number
        inv, _ = corregir_areas(carpeta, inv, salida)
    if solo_bf is None:
        solo_bf = not hay_pi(inv)
    if solo_bf:
        inv['canal'] = 'BF'
        print(f'{len(inv)} images | bright field only (no PI images): no PI stack is made')
    else:
        print(f'{len(inv)} images | bright-field/fluorescence brightness threshold = {umbral:.0f} '
              f'| BF: {(inv.canal == "BF").sum()}, fluorescence: {(inv.canal != "BF").sum()}')
    resumenes = []
    for pos, inv_pos in inv.groupby('pos'):
        if posiciones and pos not in posiciones:
            continue
        if len(inv_pos) < POS_MIN_IMAGENES:
            print(f'Position {pos}: only {len(inv_pos)} images, ignored')
            continue
        series = separar_series(carpeta, inv_pos) if solo_bf else [inv_pos]
        for inv_s in series:
            if len(inv_s) < POS_MIN_IMAGENES:
                print(f'Position {inv_s.pos.iloc[0]}: only {len(inv_s)} images, ignored'); continue
            resumenes.append(generar_posicion(carpeta, inv_s, salida, alinear, zona, margen, bf_8bits, solo_bf,
                                              fluos[0] if len(fluos) == 1 else fluos))
    resumen = pd.DataFrame(resumenes)
    if not resumen.empty:
        f = salida / 'resumen_stacks.csv'
        if f.exists() and posiciones:      # update only the processed positions
            viejo = pd.read_csv(f, dtype={'pos': str})
            resumen = pd.concat([viejo[~viejo.pos.isin(resumen.pos)], resumen]).sort_values('pos')
        resumen.to_csv(f, index=False)
        print(f'Saved to {salida}')
    return resumen


# Only from the terminal (inside Jupyter, sys.argv holds the kernel's arguments)
if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    p = argparse.ArgumentParser(description='Builds paired and aligned BF and PI stacks.')
    p.add_argument('carpeta', help='folder with the individual images')
    p.add_argument('--salida', help='output folder (default ../Stacks)')
    p.add_argument('--posiciones', nargs='+', help='only these positions (e.g. 01 02)')
    p.add_argument('--sin-alinear', action='store_true', help='do not align')
    p.add_argument('--bf-16bits', action='store_true', help='save the bright field as 16-bit (default 8-bit)')
    p.add_argument('--margen', type=int, default=MARGEN_STACKS,
                   help=f'maximum shift searched (px, default {MARGEN_STACKS})')
    p.add_argument('--solo-bf', action='store_true',
                   help='all images are bright field (no PI); detected automatically by default')
    p.add_argument('--fluo', nargs='+', default=['PI'],
                   help='fluorescence name(s) (PI, GFP...; default PI). Several: from brightest to dimmest, e.g. GFP PI')
    p.add_argument('--sin-corregir-areas', action='store_true', help='do not look for images with a shifted area number')
    a = p.parse_args()
    generar_stacks(a.carpeta, a.salida, a.posiciones, alinear=not a.sin_alinear, margen=a.margen,
                   bf_8bits=not a.bf_16bits, solo_bf=True if a.solo_bf else None, fluo=a.fluo[0] if len(a.fluo) == 1 else a.fluo,
                   corregir=not a.sin_corregir_areas)
