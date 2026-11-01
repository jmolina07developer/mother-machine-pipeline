"""
Cell detector for the straightened channels (canales_enderezados.py), trained on your
boxes (Cajas_Canales.ijm), plus mother + first daughter using the physics of the channel.

1) Each row of each frame is classified as background/gap, cell interior, upper pole or
   lower pole (gradient boosting). It uses the frame and its two neighbours: what moves between
   frames is a cell; what stays still is dirt or marks on the chip.
2) In each frame, the rows are split into cells by dynamic programming (a cell with a
   pole in the middle is two cells) and the poles are refined to the probability maximum (subpixel).
3) Mother and daughter are tracked consistently over time (segmentar_kimogramas.py,
   PARAM_DETECTOR): they grow, divide in half, and a division is forever (two cells
   cannot merge again); a bad frame can be skipped instead of inventing a division.
   Each state is scored row by row with the detector probabilities: a border between two
   cells is the lower pole of the upper cell followed by the upper pole of the lower one.
4) The mother and daughter boxes come from that tracking (poles refined in each frame); those
   further down, from step 2. With their confidence: probability of each division and uncertainty
   (px) of the borders; anything doubtful is flagged for review.

Usage:
    python detectar_celulas.py entrenar ../October/2/Stacks/Canales/*_cajas.zip
    python detectar_celulas.py validar  ../October/2/Stacks/Canales/*_cajas.zip
    python detectar_celulas.py probar   ../October/2/Stacks/Canales/*_cajas.zip ../October/9/Stacks/Canales/*_cajas.zip
    python detectar_celulas.py predecir ../October/2/Stacks/Canales/12-02102026_canales_BF.tif
    python detectar_celulas.py figura   ../October/9/Stacks/Canales/06-09102026_canales_BF.tif --canal 3 --fotos 60-100

When predicting, it saves next to the channel image:
    <image>_detector.txt           proposed boxes for Cajas_Canales.ijm ([8]):
                                   channel frame y0 y1 type doubtful (type: madre, hija or resto)
    <image>_detector_cajas.csv     all boxes: canal, foto, orden, celula, fila_sup, fila_inf,
                                   longitud_px, sale_del_canal, sd_px, dudosa
    <image>_detector_madre_hija.csv  per channel and frame: mother and daughter lengths (from their
                                   boxes), divisions, division probability, uncertainty, skipped frame
    <image>_detector_ciclos.csv    mother cycles (L0, Lc, division time, rate, partition) and
                                   the confidence of the divisions at which each cycle starts and ends
    <image>_detector_divisiones.csv  each mother and daughter division, its confidence and whether to review it
    <image>_revisar.txt            frames worth reviewing (Cajas_Canales.ijm, [9])
    <image>_linaje.csv             all cells that can be tracked (linaje.py): per cell and frame,
                                   cell number, order in the channel, poles, length
    <image>_linaje_ciclos.csv      one row per cell: mother, generation, birth (frame, min), divides or
                                   exits, duration, L0, Ld, rate, partition and whether it is born from a
                                   symmetric, polar or fragmentation division
    <image>_crecimiento_posicion.csv  expected growth per minute in each frame (all channels)

Training: your boxes (proposals you accepted also count), the proposals you
deleted with [2] (<image>_borradas.txt: there is no cell there, triple weight) and, below your last
box in each frame, background with half weight (it must not propose cells that cannot be seen with
certainty). Channels in <folder>/conjunto_prueba.csv are not used for training: `probar` uses them.
When proposing, mother/daughter boxes whose interior does not look like a cell (empty channel, gap
at the top) are removed, as are boxes further down where the detector is not confident (CONF_RESTO).

Isolated defective frames (out of focus, shifted…): mark them with [6] in Cajas_Canales.ijm; the
tracking skips them (mother and daughter = those of the previous frame, grown) and no boxes are proposed.
Unusable frames (e.g. at the end of October/2 the medium ran out and the cells burst):
<channels folder>/fotos_validas.csv with columns imagen, ultima_foto, motivo. Anything after that is
used neither for training nor for prediction.

Requires: numpy pandas scipy scikit-learn joblib tifffile roifile
"""
import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leer_cajas import leer_cajas, fotos_malas, malas_del_canal, leer_borradas  # noqa: E402

MODELO = Path(__file__).resolve().parent / 'modelos' / 'detector_canales.joblib'

ESC = [1.0, 2.0, 4.0]


def tiles(img, k, ancho, sep, T):
    """(T, L, ancho) of the channel at position k (0 = first slice)."""
    paso = ancho + sep
    return np.stack([img[k, :, t * paso:t * paso + ancho] for t in range(T)]).astype(np.float32)


def perfiles(tile):
    """Profiles along the channel (L, n) from the straightened image (L, W)."""
    W = tile.shape[1]
    t = tile / (np.median(tile) + 1e-6)
    c0, c1 = int(W * 0.3), int(W * 0.7) + 1                         # centre of the channel
    centro = t[:, c0:c1]
    borde = np.concatenate([t[:, int(W * 0.12):c0], t[:, c1:int(W * 0.88)]], 1)
    gx = np.abs(np.diff(t, axis=1))[:, int(W * 0.12):int(W * 0.88)]
    return np.stack([centro.mean(1), centro.min(1), centro.std(1), borde.mean(1),
                     centro.mean(1) - borde.mean(1), gx.max(1), t.mean(1)], 1)


def caracteristicas(T3, fila_fondo, fila_salida):
    """T3: channel tiles (T, L, W). Returns (T, L, F)."""
    Tn, L, W = T3.shape
    P = np.stack([perfiles(T3[t]) for t in range(Tn)])               # (T, L, n)
    out = []
    for t in range(Tn):
        fs = []
        for dt in (0, -1, 1):                                       # current frame and its neighbours
            u = min(max(t + dt, 0), Tn - 1)
            for j in range(P.shape[2]):
                x = P[u, :, j]
                for s in ESC:
                    g = gaussian_filter1d(x, s)
                    d1 = np.gradient(g)
                    fs += [g, d1, np.gradient(d1)] if dt == 0 else [g, d1]
        # change relative to the neighbouring frames (what moves is a cell; what stays still, chip/dirt)
        for dt in (-1, 1):
            u = min(max(t + dt, 0), Tn - 1)
            fs.append(gaussian_filter1d(P[t, :, 0] - P[u, :, 0], 1.5))
            fs.append(gaussian_filter1d(np.abs(P[t, :, 0] - P[u, :, 0]), 3.0))
        r = np.arange(L, dtype=np.float32)
        fs += [r - fila_fondo, r - fila_salida]
        out.append(np.stack(fs, 1))
    return np.stack(out).astype(np.float32)


def etiquetas(cajas_foto, L, borradas=()):
    """Class per row: 0 background/gap, 1 cell, 2 upper pole, 3 lower pole; -1 = not annotated.
    borradas: [(y0, y1)] of proposals you deleted: those rows are background even if they lie below
    your last box (unless one of your boxes covers them)."""
    y = np.zeros(L, np.int8)
    if len(cajas_foto) == 0:
        return y
    zona = int(np.ceil(cajas_foto.fila_inf.max() + 0.5)) + 3      # below this, not annotated
    for _, b in cajas_foto.iterrows():
        y0, y1 = int(round(b.fila_sup + 0.5)), int(round(b.fila_inf + 0.5))
        y[max(y0, 0):min(y1, L)] = 1
    for _, b in cajas_foto.iterrows():
        y0, y1 = int(round(b.fila_sup + 0.5)), int(round(b.fila_inf + 0.5))
        y[max(y0 - 1, 0):min(y0 + 2, L)] = 2
        if y1 < L - 1:                                              # cut off by the crop: no pole
            y[max(y1 - 2, 0):min(y1 + 1, L)] = 3
    if zona < L:
        y[zona:] = -1
    cubiertas = np.zeros(L, bool)
    for _, b in cajas_foto.iterrows():
        cubiertas[max(int(round(b.fila_sup + 0.5)) - 1, 0):min(int(round(b.fila_inf + 0.5)) + 1, L)] = True
    for y0, y1 in borradas:
        r = np.arange(max(int(round(y0)), 0), min(int(round(y1)), L))
        r = r[~cubiertas[r]]
        y[r] = 0
    return y


def filas_borradas(borradas, L):
    """Mask of the rows of deleted proposals (to give them more weight in training)."""
    m = np.zeros(L, bool)
    for y0, y1 in borradas:
        m[max(int(round(y0)), 0):min(int(round(y1)), L)] = True
    return m


def conjunto_prueba(ruta_zip):
    """Channels in the fixed test set of this image (<folder>/conjunto_prueba.csv: imagen,
    canal). They are not used for training; they are used to measure how well the detector works."""
    f = Path(ruta_zip).parent / 'conjunto_prueba.csv'
    if not f.exists(): return set()
    t = pd.read_csv(f, dtype={'imagen': str})
    base = Path(ruta_zip).name.replace('_canales_BF_cajas.zip', '').replace('_canales_BF.tif', '')
    return set(t.canal[t.imagen == base].astype(int))


def decodificar(p, min_len=6, pen=1.0, w_polo=1.0):
    """
    p: (L, 4) probabilities (background, cell, upper pole, lower pole). Splits the rows into
    background and cells (segments [j, i)) by dynamic programming. Within a cell, the central
    rows must be 'cell' (if there is a pole in the middle, they are two cells); the two rows
    at each end may be pole. Returns a list of (j, i).
    """
    L = len(p)
    eps = 1e-4
    c_bg = -np.log(p[:, 0] + p[:, 2] + p[:, 3] + eps)
    c_in = -np.log(p[:, 1] + eps)
    c_ts = -np.log(p[:, 1] + p[:, 2] + eps)          # upper end
    c_bs = -np.log(p[:, 1] + p[:, 3] + eps)          # lower end
    c_top = -np.log(p[:, 2] + eps); c_bot = -np.log(p[:, 3] + eps)
    S_in = np.concatenate([[0], np.cumsum(c_in)])
    best = np.full(L + 1, np.inf); best[0] = 0; arg = np.zeros(L + 1, int); tipo = np.zeros(L + 1, int)
    for i in range(1, L + 1):
        b = best[i - 1] + c_bg[i - 1]; a, ti = i - 1, 0
        js = np.arange(0, i - min_len + 1)
        if len(js):
            fin = i < L                                             # a cell may end at the edge
            bot = (c_bot[i - 1] * w_polo + c_bs[i - 1] + c_bs[i - 2]) if fin else (c_in[i - 1] + c_in[i - 2])
            cost = (best[js] + (S_in[i - 2] - S_in[js + 2]) + c_ts[js] + c_ts[js + 1]
                    + w_polo * c_top[js] + bot + pen)
            k = int(np.argmin(cost))
            if cost[k] < b: b, a, ti = cost[k], js[k], 1
        best[i], arg[i], tipo[i] = b, a, ti
    segs, i = [], L
    while i > 0:
        if tipo[i] == 1: segs.append((arg[i], i))
        i = arg[i]
    return segs[::-1]


def refinar(segs, p, rad=2):
    """Poles of each cell at the (subpixel) maximum of the pole probability near each end.
    Returns [(sup, inf)] in border coordinates (fila_sup + 0.5, fila_inf + 0.5)."""
    L = len(p); out = []
    def pico(q, c):
        lo, hi = max(c - rad, 0), min(c + rad + 1, L)
        k = lo + int(np.argmax(q[lo:hi]))
        if 0 < k < L - 1:
            a, b, d = q[k - 1], q[k], q[k + 1]; den = a - 2 * b + d
            return k + (0.5 * (a - d) / den if den < 0 else 0.0)
        return float(k)
    for j, i in segs:
        sup = pico(p[:, 2], j)                     # row with the highest upper-pole prob. = first row of the cell
        inf = pico(p[:, 3], i - 1) + 1 if i < L else float(i)   # last row of the cell + 1
        out.append((sup, inf))
    return out


def mapa_bordes(P, w=2):
    """
    Map of borders between cells (rows x frames) from the detector probabilities
    P (frames, rows, 4): at a border, the lower pole of the upper cell is followed by the upper
    pole of the lower one (or by empty background, if it is the last cell). A static mark inside a
    long cell usually has only one of the two.
    """
    from scipy.ndimage import maximum_filter1d
    bot = maximum_filter1d(P[:, :, 3], w + 1, axis=1, origin=w // 2)          # max in [r-w, r]
    top = maximum_filter1d(P[:, :, 2], w + 1, axis=1, origin=-(w // 2))       # max in [r, r+w]
    bg = maximum_filter1d(P[:, :, 0], w + 2, axis=1, origin=-((w + 2) // 2))
    return np.sqrt(bot * np.maximum(top, bg)).T


def _pico(q, lo, hi):
    """(Subpixel) maximum of q between rows lo and hi."""
    L = len(q); lo, hi = max(int(np.floor(lo)), 0), min(int(np.ceil(hi)) + 1, L)
    if hi <= lo: return np.nan
    k = lo + int(np.argmax(q[lo:hi]))
    if 0 < k < L - 1:
        a, b, d = q[k - 1], q[k], q[k + 1]; den = a - 2 * b + d
        return k + (0.5 * (a - d) / den if den < 0 else 0.0)
    return float(k)


def cajas_madre_hija(P, mh, fila_salida, segs):
    """
    Boxes of each frame [(sup, inf, tipo)] in border coordinates (as in refinar): mother and
    daughter come from the consistent tracking (mh), with the poles refined on the detector's pole
    probability; the cells further down, from the frame-by-frame detector (segs[t] = [(sup, inf)]).
    If the daughter is not visible (bright zone next to the exit) its box is not added.
    """
    T = P.shape[0]; out = []
    for t in range(T):
        p = P[t]; cj = []
        polo, b1, b2 = mh['polo'][t], mh['b1'][t], mh['b2'][t]
        sup_m = _pico(p[:, 2], polo - 3, polo + 3)
        if np.isnan(sup_m): sup_m = float(polo)
        def del_detector(sup):            # the detector segment that starts there
            c = [s for s in segs[t] if abs(s[0] - sup) <= 4]
            return c[0][1] if c else None
        if np.isnan(b1):                                     # the mother extends past the exit
            inf = del_detector(sup_m)
            inf = inf if inf is not None and inf - 0.5 >= fila_salida - 1 else fila_salida + 2.0
            cj.append((sup_m, inf, 'madre')); fin = inf
        else:
            inf_m = _pico(p[:, 3], b1 - 4, b1) + 1
            sup_d = _pico(p[:, 2], b1 - 1, b1 + 3)
            if np.isnan(inf_m): inf_m = float(b1)
            if np.isnan(sup_d): sup_d = float(b1)
            if inf_m - sup_m < 4: sup_m, inf_m = float(polo), float(b1)      # poorly refined poles: use the tracked borders
            cj.append((sup_m, inf_m, 'madre')); fin = inf_m
            if np.isnan(b2):
                inf = del_detector(sup_d)
                if inf is not None:
                    cj.append((sup_d, inf, 'hija')); fin = inf
            else:
                inf_d = _pico(p[:, 3], b2 - 4, b2) + 1
                if np.isnan(inf_d) or inf_d - sup_d < 4: sup_d, inf_d = float(b1), float(b2)
                cj.append((sup_d, inf_d, 'hija')); fin = inf_d
        for a, b in segs[t]:
            # below the mother and the daughter, and not outside the channel (anything starting past
            # the exit is in the main channel: it is not a cell of this channel)
            if a >= fin - 1 and a - 0.5 < fila_salida - 1: cj.append((a, b, 'resto'))
        out.append(cj)
    return out


# =========================================================================== data
def abrir(ruta_img):
    """Channel image + geometry."""
    ruta_img = Path(ruta_img)
    img = tifffile.imread(ruta_img)
    geo = pd.read_csv(ruta_img.with_name(ruta_img.stem + '_geometria.csv'))
    return img, geo


def ultima_foto(ruta_img):
    """
    Last valid frame of an image, according to <folder>/fotos_validas.csv (columns imagen,
    ultima_foto, motivo; imagen = <base> or <base>_canales_BF.tif). No file or no row: all frames.
    Later frames are used neither for training nor for prediction.
    """
    ruta_img = Path(ruta_img)
    f = ruta_img.parent / 'fotos_validas.csv'
    if not f.exists(): return None
    t = pd.read_csv(f, dtype={'imagen': str})
    base = ruta_img.name.replace('_canales_BF.tif', '').replace('_canales_BF_cajas.zip', '')
    m = t[t.imagen.astype(str).str.replace('_canales_BF.tif', '', regex=False) == base]
    return int(m.ultima_foto.iloc[0]) if len(m) else None


def caracteristicas_canal(img, geo, k, n_fotos=None):
    g = geo.iloc[k]
    T = int(g.fotos) if n_fotos is None else min(int(n_fotos), int(g.fotos))
    return caracteristicas(tiles(img, k, int(g.ancho), int(g.sep), T), g.fila_fondo, g.fila_salida)


PESO_BORRADAS = 3.0     # rows of proposals you deleted count triple in training
PESO_ABAJO = 0.5        # below your last box (where you stop seeing cells with certainty) = background, with half weight


def datos_entrenamiento(rutas_cajas, prueba='excluir'):
    """
    List of annotated frames: dict(base, canal, foto, X (L, F), y (L,), w (L,), cajas, prueba).
    prueba: 'excluir' (does not use the test-set channels), 'solo' (only those) or 'todos'.
    """
    datos = []
    for rz in rutas_cajas:
        rz = Path(rz)
        ruta_img = rz.with_name(rz.name.replace('_cajas.zip', '.tif'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cajas, vacias, anot = leer_cajas(rz)
        if not len(anot):
            continue
        img, geo = abrir(ruta_img)
        uf = ultima_foto(ruta_img)
        if uf is not None:
            anot = anot[anot.foto <= uf]; cajas = cajas[cajas.foto <= uf]
        malas = fotos_malas(rz)                              # defective frames: not used
        if len(malas):
            es_mala = np.array([f in set(malas_del_canal(malas, c)) for c, f in zip(anot.canal, anot.foto)], bool)
            anot = anot[~es_mala]
        test = conjunto_prueba(rz)
        if prueba == 'excluir': anot = anot[~anot.canal.isin(test)]
        elif prueba == 'solo': anot = anot[anot.canal.isin(test)]
        borr = leer_borradas(rz)
        for c in sorted(anot.canal.unique()):
            k = int(np.where(geo.canal.values == c)[0][0])
            F = caracteristicas_canal(img, geo, k, uf)
            for f in sorted(anot[anot.canal == c].foto.unique()):
                cf = cajas[(cajas.canal == c) & (cajas.foto == f)]
                bb = [(r.y0, r.y1) for r in borr[(borr.canal == c) & (borr.foto == f)].itertuples()]
                w = np.where(filas_borradas(bb, F.shape[1]), PESO_BORRADAS, 1.0)
                y = etiquetas(cf, F.shape[1], bb)
                # below your last box there is no cell that can be seen with certainty: the
                # detector must not propose boxes there (background, with half weight)
                abajo = y < 0; y[abajo] = 0; w[abajo] = PESO_ABAJO
                datos.append(dict(base=rz.name.replace('_canales_BF_cajas.zip', ''), canal=int(c), foto=int(f),
                                  X=F[f - 1].copy(), y=y, w=w, cajas=cf, borradas=bb, prueba=int(c) in test, F=None))
        nb = int(sum(len(d['borradas']) for d in datos if d['base'] == rz.name.replace('_canales_BF_cajas.zip', '')))
        print(f'  {rz.name}: {len(anot)} annotated frames in {anot.canal.nunique()} channels, {len(cajas)} boxes'
              + (f', {nb} deleted proposals' if nb else ''))
    return datos


def _entrenar_clf(datos):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.concatenate([d['X'][d['y'] >= 0] for d in datos])
    y = np.concatenate([d['y'][d['y'] >= 0] for d in datos])
    extra = np.concatenate([d.get('w', np.ones(len(d['y'])))[d['y'] >= 0] for d in datos])
    w = np.ones(len(y)); cnt = np.bincount(y, minlength=4)
    for c in range(4): w[y == c] = len(y) / (4 * max(cnt[c], 1))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          random_state=0).fit(X, y, sample_weight=np.sqrt(w) * extra)


def _iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0


def _evaluar(datos, probs):
    from scipy.optimize import linear_sum_assignment
    filas = []
    for d, p in zip(datos, probs):
        segs = refinar(decodificar(p), p)
        verd = [(r.fila_sup + 0.5, r.fila_inf + 0.5, r.sale_del_canal) for _, r in d['cajas'].sort_values('fila_sup').iterrows()]
        zona = max([v[1] for v in verd], default=0) + 3
        pr = [s for s in segs if s[0] < zona]
        tp = 0; ep = []
        if verd and pr:
            M = np.array([[_iou(v, q) for q in pr] for v in verd]); a_, b_ = linear_sum_assignment(-M)
            for a, b in zip(a_, b_):
                if M[a, b] >= 0.5:
                    tp += 1; ep.append(abs(verd[a][0] - pr[b][0]))
                    if not verd[a][2]: ep.append(abs(verd[a][1] - pr[b][1]))
        filas.append(dict(base=d['base'], n_v=len(verd), n_p=len(pr), tp=tp, ep=np.median(ep) if ep else np.nan))
    R = pd.DataFrame(filas)
    return R.groupby('base').apply(lambda g: pd.Series(dict(
        fotos=len(g), cajas=int(g.n_v.sum()), precision=g.tp.sum() / max(g.n_p.sum(), 1),
        recall=g.tp.sum() / max(g.n_v.sum(), 1), n_exacto=float(np.mean(g.n_p == g.n_v)),
        error_polos_px=float(np.nanmedian(g.ep))))).round(3)


# =========================================================================== commands
def entrenar(rutas_cajas, ruta_modelo=MODELO):
    import joblib
    t0 = time.time()
    datos = datos_entrenamiento(rutas_cajas)
    clf = _entrenar_clf(datos)
    ruta_modelo = Path(ruta_modelo); ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(fecha=datetime.now().strftime('%Y-%m-%d %H:%M'), cajas=[Path(r).name for r in rutas_cajas],
                fotos=len(datos), n_cajas=int(sum(len(d['cajas']) for d in datos)))
    joblib.dump(dict(clf=clf, meta=meta), ruta_modelo)
    ruta_modelo.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f'Saved: {ruta_modelo} ({len(datos)} frames, {time.time() - t0:.0f} s)')


def probar(rutas_cajas, modelo):
    """
    Evaluates the model on the fixed test set (conjunto_prueba.csv), frame by frame: precision and
    recall of the boxes (overlap ≥ 50 %) and invented boxes per 100 frames (a box that barely
    touches any of yours, above your last box or where you deleted a proposal).
    """
    from scipy.optimize import linear_sum_assignment
    datos = datos_entrenamiento(rutas_cajas, prueba='solo')
    if not datos:
        print('No frames from the test set (conjunto_prueba.csv).'); return
    filas = []
    for d in datos:
        p = modelo['clf'].predict_proba(d['X'])
        pr = []
        for a, b in refinar(decodificar(p), p):
            lo, hi = int(round(a)) + 2, int(round(b)) - 2
            if (p[lo:hi, 1].mean() if hi > lo else 1.0) >= CONF_RESTO: pr.append((a, b))
        verd = [(r.fila_sup + 0.5, r.fila_inf + 0.5) for _, r in d['cajas'].sort_values('fila_sup').iterrows()]
        zona = max([v[1] for v in verd]) + 3 if verd else np.inf
        inv = 0
        for a, b in pr:
            cub = sum(max(0, min(b, v[1]) - max(a, v[0])) for v in verd) / max(b - a, 1e-6)
            if cub < 0.3 and (a < zona or any(min(b, y1) - max(a, y0) > 0.5 * (b - a) for y0, y1 in d['borradas'])): inv += 1
        pz = [s for s in pr if s[0] < zona]; tp = 0
        if verd and pz:
            M = np.array([[_iou(v, q) for q in pz] for v in verd]); a_, b_ = linear_sum_assignment(-M)
            tp = int(sum(M[i, j] >= 0.5 for i, j in zip(a_, b_)))
        filas.append(dict(base=d['base'], nv=len(verd), np_=len(pz), tp=tp, inv=inv))
    R = pd.DataFrame(filas)
    res = R.groupby('base').apply(lambda g: pd.Series(dict(
        fotos=len(g), precision=g.tp.sum() / max(g.np_.sum(), 1), recall=g.tp.sum() / max(g.nv.sum(), 1),
        inventadas_por_100_fotos=100 * g.inv.sum() / len(g)))).round(3)
    print(res.to_string())
    return res


def validar(rutas_cajas, pliegues=5):
    """Cross-validation by channel: each channel is predicted by a model that has not seen it."""
    datos = datos_entrenamiento(rutas_cajas)
    rng = np.random.default_rng(0); grupo = {}
    for d in datos:
        grupo.setdefault((d['base'], d['canal']), rng.integers(pliegues))
    fold = np.array([grupo[(d['base'], d['canal'])] for d in datos])
    probs = [None] * len(datos)
    for k in range(pliegues):
        clf = _entrenar_clf([d for d, f in zip(datos, fold) if f != k])
        for i, (d, f) in enumerate(zip(datos, fold)):
            if f == k: probs[i] = clf.predict_proba(d['X'])
        print(f'  fold {k + 1}/{pliegues}')
    res = _evaluar(datos, probs)
    print(res.to_string())
    return res


UMBRAL_SD = 3.0          # px: border uncertainty above which the frame is flagged for review
MAX_SOLAPE = 2.0         # px: two proposed boxes cannot overlap by more than this
HUECO_ARRIBA = True      # the mother may start below the closed end if there is an empty gap above it


def sin_solapes(cajas):
    """Sorted boxes (sup, inf, tipo) without overlaps: if two overlap slightly (≤ 4 px) the border is
    placed halfway; if one lies inside the other or they overlap a lot, the one from the frame-by-frame
    detector ('resto') is removed or, if neither is, the lower one."""
    out = []
    for a, b, tp in cajas:
        if out:
            a0, b0, tp0 = out[-1]
            sol = min(b, b0) - max(a, a0)
            if sol > MAX_SOLAPE:
                if sol <= 4 and b > b0 and a > a0:
                    m = 0.5 * (a + b0); out[-1] = (a0, m, tp0); a = m
                else:
                    if tp0 == 'resto' and tp != 'resto': out[-1] = (a, b, tp)
                    continue
        out.append((a, b, tp))
    return out
PESO_POLO_SEGUIMIENTO = 1.0   # 0.5 brought divisions forward less in LB 25/09 but delayed them in October/9 (13/10):
                              # in the tracking the poles weigh the same as in the boxes (1.0)
def _p_seguimiento(P):
    Q = P * np.array([1.0, 1.0, PESO_POLO_SEGUIMIENTO, PESO_POLO_SEGUIMIENTO], np.float32)
    return Q / Q.sum(2, keepdims=True)
UNIR_RESTO = False
UNIR_MAX_TROZO = 15.0    # only if one of the two pieces is shorter than this (px)
def _bordes_cascada(cs, t):
    """Borders between cells tracked by the cascade in frame t (top to bottom, up to the first one not visible)."""
    out = []
    for k in range(1, max(cs['B']) + 1):
        y = cs['B'][k][t]
        if np.isnan(y): break
        out.append(float(y))
    return out
def unir_resto(cajas, bordes, margen=4.0):
    """Two consecutive 'resto' boxes (from the frame-by-frame detector) are merged if, in the zone
    tracked by the cascade (above its last border), there is no border between them: the detector has
    split a cell that the time-consistent tracking sees as a whole (e.g. a septum that has not yet
    separated the cells)."""
    if not bordes: return cajas
    lim = bordes[-1]; out = []
    for a, b, tp in cajas:
        if out and tp == 'resto' and out[-1][2] == 'resto':
            a0, b0, _ = out[-1]
            if b0 < lim - margen and min(b0 - a0, b - a) < UNIR_MAX_TROZO and not any(b0 - margen <= y <= a + margen for y in bordes):
                out[-1] = (a0, b, 'resto'); continue
        out.append((a, b, tp))
    return out
CONF_MADRE_HIJA = 0.5    # a mother or daughter box whose interior does not look like a cell (empty channel, gap at the top) is removed
CONF_RESTO = 0.8         # boxes further down (frame by frame) are only proposed if the detector is confident
UMBRAL_CONF = 0.8        # minimum confidence of a division for it not to be flagged


def _canal(P, g, tiempo, malas=(), tasa=None, prm=None, linaje=True, conf_resto=None, cajas_red=None, mu=None,
           manuales=None):
    """Consistent mother and daughter, boxes, divisions and frames to review for one channel.
    malas: frames (1-based) marked as defective: the tracking skips them and they get no boxes."""
    import segmentar_kimogramas as SK
    T = P.shape[0]; c = int(g.canal); fs = float(g.fila_salida)
    mala = np.zeros(T, bool); mala[[f - 1 for f in malas if 1 <= f <= T]] = True
    segs = [[] if mala[t] else refinar(decodificar(P[t]), P[t]) for t in range(T)]
    Pt = _p_seguimiento(P)
    Ps = (Pt[:, :, 2] + Pt[:, :, 3]).T
    prm = prm or SK.PARAM_DETECTOR
    # gap at the top: if the detector sees empty background between the closed end and the first cell,
    # the mother (the cell closest to the closed end) starts there, not at the closed end
    polo_min = np.full(T, np.nan)
    if HUECO_ARRIBA:
        for t in range(T):
            if segs[t]:
                a = segs[t][0][0]; lo, hi = int(g.fila_fondo) + 2, int(a) - 1
                if a > g.fila_fondo + 8 and hi > lo and P[t, lo:hi, 0].mean() > 0.6:
                    polo_min[t] = a
    _, mh = SK.segmentar_canal(Ps, g.fila_fondo, fs, prm_madre=prm, temp=SK.TEMPERATURA,
                               P4=Pt, P_bordes=mapa_bordes(Pt), malas=list(np.where(mala)[0]), tasa=tasa,
                               polo_min=polo_min)
    cj = cajas_madre_hija(P, mh, fs, segs)
    cs = None
    if linaje:                                               # all cells that can be tracked
        import linaje as LN
        cs = LN.cascada(Pt, mh, fs, prm, max_bordes=MAX_BORDES)
    def conf(p, a, b):                                    # mean 'cell' probability in the interior
        lo, hi = int(round(a)) + 2, int(round(b)) - 2
        return float(p[lo:hi, 1].mean()) if hi > lo else float(p[int(round(a)):max(int(round(b)), int(round(a)) + 1), 1:].sum(1).mean())
    no_vis = np.zeros(T, bool)
    for t in range(T):
        coh = [(a, b, tp) for a, b, tp in cj[t] if tp != 'resto' and conf(P[t], a, b) >= CONF_MADRE_HIJA]
        if not any(tp == 'madre' for _, _, tp in coh):
            no_vis[t] = True; coh = []                   # no visible mother: no daughter either
        # the rest, from the frame-by-frame detector: what does not overlap mother and daughter, inside the channel and confident
        resto = []
        for a, b in segs[t]:
            sol = max([max(0, min(b, y) - max(a, x)) for x, y, _ in coh] + [0]) / max(b - a, 1e-6)
            sol_px = max([max(0, min(b, y) - max(a, x)) for x, y, _ in coh] + [0])
            if sol_px <= MAX_SOLAPE and a - 0.5 < fs - 1 and conf(P[t], a, b) >= (conf_resto or CONF_RESTO):
                resto.append((a, b, 'resto'))
        cj[t] = sin_solapes(sorted(coh + resto))
        if cs is not None and UNIR_RESTO:
            cj[t] = unir_resto(cj[t], _bordes_cascada(cs, t))
    md = SK.tabla_madre_hija(mh, c, tiempo)
    sd1 = np.nan_to_num(mh['sd_b1'], nan=0.0); sd2 = np.nan_to_num(mh['sd_b2'], nan=0.0)
    dud_m = (sd1 > UMBRAL_SD) | mh['saltado']; dud_h = (sd2 > UMBRAL_SD) | mh['saltado']
    cajas, Lm, Lh, sm, sh = [], np.full(T, np.nan), np.full(T, np.nan), np.ones(T, bool), np.ones(T, bool)
    for t in range(T):
        if mala[t]: continue                                 # defective frame: no boxes
        for n, (a, b, tipo) in enumerate(cj[t]):
            sale = bool(b - 0.5 >= fs - 1)
            dud = bool(dud_m[t]) if tipo == 'madre' else (bool(dud_h[t]) if tipo == 'hija' else False)
            sd = sd1[t] if tipo == 'madre' else (sd2[t] if tipo == 'hija' else np.nan)
            cajas.append(dict(canal=c, foto=t + 1, orden=n + 1, celula=tipo, fila_sup=a - 0.5, fila_inf=b - 0.5,
                              longitud_px=b - a, sale_del_canal=sale, sd_px=round(float(sd), 2), dudosa=dud))
            if tipo == 'madre': Lm[t] = np.nan if sale else b - a; sm[t] = sale
            if tipo == 'hija': Lh[t] = np.nan if sale else b - a; sh[t] = sale
    sm[mala] = False; sh[mala] = False                       # defective frame: no measurement (not 'out')
    sm[no_vis] = False; sh[no_vis] = False                   # no visible mother: not 'out' either
    # box lengths (same convention as your boxes: pole to pole)
    md['L_madre_px'] = Lm; md['L_hija_px'] = Lh; md['madre_fuera'] = sm; md['hija_fuera'] = sh
    md['dudosa_madre'] = dud_m; md['dudosa_hija'] = dud_h
    md['foto_mala'] = mala                                   # marked by hand as defective
    md['madre_no_visible'] = no_vis & ~mala                  # empty channel or gap at the top: no mother box
    # divisions with their confidence (probability that there is a division in that frame ±1)
    divs, revisar = [], []
    for cel, dk, pk, sd in (('madre', 'div_madre', 'p_div_madre', sd1), ('hija', 'div_hija', 'p_div_hija', sd2)):
        p = mh[pk]; w = np.array([p[max(0, t - 1):t + 2].sum() for t in range(T)]).clip(0, 1)
        ev = set(mh[dk])
        for t in sorted(ev):
            conf = float(w[t]); inc = float(sd[max(0, t - 1):t + 2].max())
            rev = conf < UMBRAL_CONF          # (at the division itself the border is always somewhat uncertain)
            divs.append(dict(canal=c, foto=t + 1, celula=cel, tipo='division', confianza=round(conf, 3),
                             incertidumbre_px=round(inc, 1), revisar=rev))
            if rev: revisar.append((c, t + 1, f'division de la {cel} dudosa (confianza {conf:.2f})'))
        # the tracking places no division but the probability is fairly high: missed division?
        for t in range(1, T):
            if w[t] >= 0.3 and w[t] == w[max(0, t - 2):t + 3].max() and not any(abs(t - e) <= 2 for e in ev):
                divs.append(dict(canal=c, foto=t + 1, celula=cel, tipo='posible (no puesta)', confianza=round(float(w[t]), 3),
                                 incertidumbre_px=round(float(sd[max(0, t - 1):t + 2].max()), 1), revisar=True))
                revisar.append((c, t + 1, f'posible division de la {cel} no puesta (probabilidad {w[t]:.2f})'))
    # runs of frames with an uncertain mother or daughter, or skipped frames (the first frame of each run)
    for nombre, m in (('madre incierta', sd1 > UMBRAL_SD), ('hija incierta', sd2 > UMBRAL_SD),
                      ('foto mala (saltada)', mh['saltado'] & ~mala)):
        t = 0
        while t < T:
            if m[t]:
                u = t
                while u + 1 < T and m[u + 1]: u += 1
                revisar.append((c, t + 1, nombre + (f' (fotos {t + 1}-{u + 1})' if u > t else '')))
                t = u + 1
            else:
                t += 1
    lin = None
    if linaje:
        borde = bool(getattr(g, 'al_borde', False))
        if cajas_red is not None:
            # with the network: tracking of its boxes allowing gaps between cells; global (the whole movie at
            # once, linaje.seguir_cajas_global) or frame by frame (linaje.seguir_cajas)
            args = (cajas_red, tiempo, float(g.fila_fondo), fs)
            if SEGUIMIENTO == 'global':
                try:
                    F, Cc = LN.seguir_cajas_global(*args, mu=mu, malas=malas, manuales=manuales)
                except Exception as e:                       # no solution or no recent scipy: frame by frame
                    print(f'  channel {c}: global tracking not available ({e}); using frame by frame')
                    F, Cc = LN.seguir_cajas(*args, mu=mu, malas=malas)
            else:
                F, Cc = LN.seguir_cajas(*args, mu=mu, malas=malas)
        else:
            # tree-based detector: cascade from the closed end (cells one after another)
            F, Cc = LN.linaje(cs, Pt, fs)
            if len(F):
                F = F[~F.foto.isin(np.where(mala)[0] + 1)]
                F = visibilidad_linaje(F, P, segs, fs, conf_resto=conf_resto)
                F['seguida'] = True
        if len(F) and cajas_red is not None and CANAL_VACIO:
            F = canal_vacio(F, Cc, float(g.fila_fondo), fs)
        if len(F):
            if borde:                                        # channel cut off by the image edge: not measured
                F['visible'] = False; F['longitud_px'] = np.nan
            lin = LN.tablas(F, Cc, c, tiempo)
    return cajas, md, divs, revisar, lin


CANAL_VACIO = True       # remove 'cells' that are actually an empty channel (see canal_vacio)


def canal_vacio(F, C, fondo, fila_salida, min_fotos=6, frac=0.7, crece_max=1.15):
    """
    The network sometimes sees a whole empty channel as one cell (KAN-TRIM LB 22/03/2024: ~40 % of the channels;
    also in Trim). A 'cell' that fills at least 70 % of the channel, is seen in at least 6 frames, does not divide and
    does not grow (median of its last 3 boxes < 1.15 × that of its first 3) is an empty channel: it is made not visible
    and has no length. New column `canal_vacio`. Real filaments grow or divide.
    """
    F = F.copy(); F['canal_vacio'] = False
    fin = dict(zip(C.celula, C.final)) if len(C) and 'final' in C else {}
    largo = fila_salida - fondo
    for cid, g in F.groupby('celula'):
        if len(g) < min_fotos or fin.get(cid) == 'division': continue
        g = g.sort_values('foto'); Lb = (g.fila_inf - g.fila_sup).to_numpy()
        if np.median(Lb) < frac * largo: continue
        if np.median(Lb[-3:]) < crece_max * max(np.median(Lb[:3]), 1.0):
            F.loc[g.index, 'canal_vacio'] = True
    F.loc[F.canal_vacio, 'visible'] = False; F.loc[F.canal_vacio, 'longitud_px'] = np.nan
    return F


MAX_BORDES = 20          # borders we try to track downwards (mother, daughter and whatever is visible)
CONF_LINAJE = 0.5        # a lineage cell is only visible (box and length) if its interior looks like a cell
CONF_ESTATICA = 0.8      # a 'cell' that does not grow over the whole movie and has a median confidence below this is not a cell
FOTOS_ESTATICA = 30


def visibilidad_linaje(F, P, segs, fila_salida, conf_resto=None, regla_estatica=True):
    """
    The cascade tracking places a cell between each pair of consecutive borders, and the last one extends
    to the channel exit. Here we check whether there really is a cell:
    - confianza = mean 'cell' probability of the detector in its interior;
    - the last one (the one that 'exits the channel') is cut where the cell seen by the detector in that
      frame ends: if it ends before the exit, its box extends only that far (fila_inf_visible);
    - visible = confianza >= CONF_LINAJE (mother and daughter) or CONF_RESTO (those further down), the same
      thresholds as for the proposed boxes. Non-visible cells have neither length nor box;
    - a cell that neither grows nor divides over the whole movie, with low median confidence, is the closed
      end of an empty channel (or dirt): not visible.
    """
    F = F.copy()
    conf, finv = [], []
    for r in F.itertuples():
        t = r.foto - 1; a = r.fila_sup + 0.5; b = r.fila_inf + 0.5
        if r.sale_del_canal:
            c_ = [sb for sa, sb in segs[t] if abs(sa - a) <= 4]
            b = min(c_[0], fila_salida - 1) if c_ else min(b, fila_salida - 1)
        lo, hi = int(round(a)) + 2, int(round(b)) - 2
        conf.append(float(P[t, lo:hi, 1].mean()) if hi > lo else 0.0); finv.append(b - 0.5)
    F['fila_inf_visible'] = np.round(finv, 2); F['confianza'] = np.round(conf, 3)
    F['visible'] = F.confianza >= np.where(F.orden <= 2, CONF_LINAJE, conf_resto or CONF_RESTO)   # like the proposed boxes
    for cid, g in (F.groupby('celula') if regla_estatica else []):
        L = g.longitud_px.dropna().to_numpy()
        if len(g) >= FOTOS_ESTATICA and g.confianza.median() < CONF_ESTATICA and len(L) >= 12:
            q = len(L) // 4                                  # does not grow: the last quarter is as long as the first
            if np.median(L[-q:]) < 1.5 * np.median(L[:q]): F.loc[g.index, 'visible'] = False
    F.loc[~F.visible, 'longitud_px'] = np.nan
    return F


def parametros_posicion(Ps_canales, segs_canales, geo, tiempo):
    """
    What is estimated from all channels of a position before tracking the cells:
    - expected growth rate from each frame to the next (actual time of each frame; it changes
      with the medium), from how much the borders move in each channel;
    - l0: typical length of a newborn cell (for the polar divisions of filaments).
    """
    import segmentar_kimogramas as SK
    lams = []
    for Ps, (_, g) in zip(Ps_canales, geo.iterrows()):
        info = {}; SK.seguir_bordes(Ps, g.fila_fondo, g.fila_salida, info=info); lams.append(info['lams'])
    tasa, mu = SK.tasa_posicion(lams, tiempo)
    L = [b - a for segs in segs_canales for a, b in segs]
    l0 = float(np.clip(np.percentile(L, 25), 8, 40)) if len(L) > 20 else 15.0
    return tasa, mu, l0


SEGUIMIENTO = 'global'       # 'global': the whole movie of each channel at once (integer programming, ~1 s per channel)
                             # 'foto_a_foto': matching each frame with the next one (faster, somewhat worse)
DETECTOR_CAJAS = 'centros'   # 'centros' (network, modelos/detector_centros.onnx): boxes for [8], tracking and lineage with the network
                             # 'arboles': everything with the tree-based detector (modelos/detector_canales.joblib)


def P_de_cajas_centros(cajas_fotos, L):
    """
    Per-row probabilities for the tracking from the boxes of the centre-based detector, as is
    done with your boxes for the reference reconstruction: inside each box, cell (and its
    poles) with probability 0.06 + 0.88 · box confidence; outside, background with 0.94.
    cajas_fotos: one list per frame of [(fila_sup, fila_inf, confianza)] -> (T, L, 4).
    """
    T = len(cajas_fotos); P = np.zeros((T, L, 4), np.float32)
    for t, cj in enumerate(cajas_fotos):
        df = pd.DataFrame([dict(fila_sup=a, fila_inf=b) for a, b, _ in cj], columns=['fila_sup', 'fila_inf'])
        y = etiquetas(df, L) if len(df) else np.zeros(L, np.int8); y = np.where(y < 0, 0, y)
        conf = np.full(L, 0.94, np.float32)
        for a, b, c in cj:
            r0, r1 = max(0, int(round(a)) - 1), min(L, int(round(b)) + 2)
            conf[r0:r1] = np.minimum(conf[r0:r1], 0.06 + 0.88 * c)
        P[t] = ((1 - conf) / 3)[:, None]; P[t, np.arange(L), y] = conf
    return P


def cajas_centros_posicion(ruta_img, img=None, geo=None):
    """
    Centre-based detector (network exported to ONNX; requires onnxruntime) on all channels of an
    image. Returns {k: dict(cajas=[[(fila_sup, fila_inf, confianza)] per frame], L=…)} or None if it
    is not available (no onnxruntime or no model).
    """
    try:
        import centros_comun as CC
        det = CC.DetectorCentros()
    except Exception as e:
        print(f'  (centre-based detector not available: {e})')
        return None
    ruta_img = Path(ruta_img)
    if img is None: img, geo = abrir(ruta_img)
    uf = ultima_foto(ruta_img); out = {}
    for k, (_, g) in enumerate(geo.iterrows()):
        T = int(g.fotos) if uf is None else min(int(uf), int(g.fotos))
        T3 = tiles(img, k, int(g.ancho), int(g.sep), T).astype(np.uint8)
        cj, (_, _, Pcl) = det.cajas(CC.recorte_fijo(T3), T3.shape[1], float(g.fila_fondo), float(g.fila_salida))
        out[k] = dict(cajas=cj, L=T3.shape[1], P=Pcl.astype(np.float32))
    return out


def escribir_propuestas_centros(ruta_img, geo, cc, revisar_previo=None):
    """<image>_detector.txt (boxes for [8]; dudosa = confianza < UMBRAL_DUDOSA), _detector_centros.csv and
    the frames with any doubtful box added to _revisar.txt (for [9])."""
    import centros_comun as CC
    ruta_img = Path(ruta_img); pref = ruta_img.with_suffix('')
    malas = fotos_malas(ruta_img.with_name(ruta_img.stem + '_cajas.zip'))
    filas, lineas, dudas = [], [], []
    for k, (_, g) in enumerate(geo.iterrows()):
        c = int(g.canal); mal = set(malas_del_canal(malas, c))
        for t, cajas_t in enumerate(cc[k]['cajas']):
            f = t + 1
            if f in mal: continue
            dud_f = []
            for n, (a, b, conf) in enumerate(cajas_t):
                if a - 0.5 >= g.fila_salida - 1: continue          # starts past the exit: main channel
                dud = conf < CC.UMBRAL_DUDOSA
                lineas.append(f'{c} {f} {a:.2f} {b:.2f} centros {int(dud)}\n')
                filas.append(dict(canal=c, foto=f, orden=n + 1, fila_sup=round(a, 2), fila_inf=round(b, 2),
                                  longitud_px=round(b - a, 2), confianza=round(conf, 3), dudosa=dud,
                                  sale_del_canal=bool(b >= g.fila_salida)))
                if dud: dud_f.append(conf)
            if dud_f: dudas.append((c, f, f'caja dudosa del detector por centros (confianza {min(dud_f):.2f})'))
    Path(f'{pref}_detector.txt').write_text(''.join(lineas))
    pd.DataFrame(filas).to_csv(f'{pref}_detector_centros.csv', index=False)
    rv = Path(f'{pref}_revisar.txt')
    previo = revisar_previo if revisar_previo is not None else (
        [tuple(l.split(' ', 2)) for l in rv.read_text().splitlines() if l.strip()] if rv.exists() else [])
    previo = [(int(c), int(f), m) for c, f, m in previo if 'detector por centros' not in m]
    todo = sorted(set(previo + dudas))
    rv.write_text(''.join(f'{c} {f} {m}\n' for c, f, m in todo))
    return len(filas), sum(r['dudosa'] for r in filas), len(dudas), len(todo)


def proponer_centros(ruta_img, revisar_previo=None):
    """Only the boxes for [8] (and their doubts in _revisar.txt) from the centre-based detector, without redoing the lineage."""
    img, geo = abrir(ruta_img)
    cc = cajas_centros_posicion(ruta_img, img, geo)
    if cc is None: return None
    n, nd, nf, _ = escribir_propuestas_centros(ruta_img, geo, cc, revisar_previo)
    print(f'  {Path(ruta_img).name}: boxes for [8] from the centre-based detector: {n} boxes, {nd} doubtful in {nf} frames')


def _dudas_canal(a):
    """Global tracking of one channel (in a separate process) -> its doubts."""
    import linaje as LN
    cajas, tiempo, fondo, salida, mu, malas = a
    info = {}
    LN.seguir_cajas_global(cajas, tiempo, fondo, salida, mu=mu, malas=malas, info=info)
    return info.get('dudas', [])


def lista_revision(ruta_img, n=20, frac_azar=0.15, semilla=0):
    """
    Frames to review with [8] and [9] where the global tracking of the network's boxes is most in doubt
    (<image>_revisar_seguimiento.txt). For each channel and frame the doubts are added up: the network misses
    a cell that was there (3), splits a cell in two (3), sees a box that the tracking discards (2), a cell
    disappears inside the channel (2), the box does not match the expected length (1), division (1). The
    n frames with the most doubts are chosen (at most 2 per channel and at least 4 frames apart), plus a
    fraction frac_azar of random frames so as not to train only on rare cases. Frames that already have
    your boxes, bad frames and channels cut off by the edge are not proposed. Also redoes the boxes for [8].
    """
    import linaje as LN
    from leer_cajas import leer_cajas
    ruta_img = Path(ruta_img)
    img, geo = abrir(ruta_img)
    cc = cajas_centros_posicion(ruta_img, img, geo)
    if cc is None: sys.exit('The centre-based detector is required (onnxruntime and modelos/detector_centros.onnx).')
    escribir_propuestas_centros(ruta_img, geo, cc)                     # boxes for [8] brought up to date
    base = ruta_img.stem.replace('_canales_BF', '')
    kimo = ruta_img.parent.parent / 'Kimogramas' / f'{base}_kimogramas.npz'
    T = len(cc[0]['cajas'])
    tiempo = (np.load(kimo, allow_pickle=True)['tiempo_min'] if kimo.exists() else np.arange(T, dtype=float) * 5.0)[:T]
    cp = Path(str(ruta_img.with_suffix('')) + '_crecimiento_posicion.csv')
    mu = pd.read_csv(cp).tasa_esperada_por_min.values[:T] if cp.exists() else None
    zp = ruta_img.with_name(ruta_img.stem + '_cajas.zip')
    malas = fotos_malas(zp) if zp.exists() else []
    anotadas = set()
    if zp.exists():
        _, _, anot = leer_cajas(zp)
        anotadas = {(int(c), int(f)) for c, f in zip(anot.canal, anot.foto)}
    punt, motivos = {}, {}
    validas = []
    trabajos = []                                                       # (channel, bad frames, tracking arguments)
    for k, (_, g) in enumerate(geo.iterrows()):
        c = int(g.canal)
        if bool(getattr(g, 'al_borde', False)): continue
        if sum(len(x) for x in cc[k]['cajas']) < T: continue           # almost empty channel
        ml = set(malas_del_canal(malas, c)) if len(malas) else set()
        trabajos.append((c, ml, (cc[k]['cajas'][:T], tiempo, float(g.fila_fondo), float(g.fila_salida), mu, sorted(ml))))
    n_proc = max(1, min(PROCESOS, (os.cpu_count() or 1), len(trabajos)))
    if n_proc > 1:                                                      # one channel per process
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_proc) as ex:
            todas = list(ex.map(_dudas_canal, [t[2] for t in trabajos]))
    else:
        todas = [_dudas_canal(t[2]) for t in trabajos]
    for (c, ml, _), dudas in zip(trabajos, todas):
        for f, w, m in dudas:
            if (c, f) in anotadas or f in ml: continue
            if m.startswith('desaparece') and f <= 3: continue           # when the chip is loaded the flow washes cells away
            punt[(c, f)] = punt.get((c, f), 0) + w
            motivos.setdefault((c, f), {}); motivos[(c, f)][m] = motivos[(c, f)].get(m, 0) + 1
        validas += [(c, f) for f in range(1, T + 1) if (c, f) not in anotadas and f not in ml]
    elegidas, por_canal = [], {}
    for (c, f), w in sorted(punt.items(), key=lambda kv: -kv[1]):
        if len(elegidas) >= n: break
        if len(por_canal.get(c, [])) >= 2 or any(abs(f - f2) < 4 for f2 in por_canal.get(c, [])): continue
        elegidas.append((c, f, 'seguimiento: ' + ', '.join(f'{m}' + (f' (×{v})' if v > 1 else '')
                                                         for m, v in sorted(motivos[(c, f)].items(), key=lambda kv: -kv[1]))))
        por_canal.setdefault(c, []).append(f)
    rng = np.random.default_rng(semilla)
    resto = [x for x in validas if x not in {(c, f) for c, f, _ in elegidas}]
    for i in rng.permutation(len(resto))[:max(1, int(round(frac_azar * n)))]:
        c, f = resto[i]; elegidas.append((c, f, 'al azar (para comparar)'))
    elegidas.sort()
    pref = ruta_img.with_suffix('')
    Path(f'{pref}_revisar_seguimiento.txt').write_text(''.join(f'{c} {f} {m}\n' for c, f, m in elegidas))
    print(f'{ruta_img.name}: {len(elegidas)} frames to review in {Path(f"{pref}_revisar_seguimiento.txt").name}')
    return elegidas


PROCESOS = 4             # channels tracked at the same time (processes)


def _canal_args(a):
    return _canal(*a)


USAR_CAJAS_MANUALES = True   # in the frames you corrected by hand (your boxes in <image>_cajas.zip) the tracking uses your boxes
ZONA_MANUAL = 3.0            # px below your last box that count as annotated (as in training)


def cajas_manuales(ruta_img, geo, T):
    """
    Your boxes (Cajas_Canales.ijm) from the frames you corrected: {canal: {t (0-based): (cajas, zona)}}, with
    cajas = [(fila_sup, fila_inf, 1.0)] in the coordinates of the network's boxes and zona = row down to which
    you annotated (3 px below your last box; the whole channel if you marked it empty). Only those frames: in
    the others the network's boxes are used. Defective frames do not count.
    """
    zp = Path(ruta_img).with_name(Path(ruta_img).stem + '_cajas.zip')
    if not USAR_CAJAS_MANUALES or not zp.exists(): return {}
    cajas, vacias, anot = leer_cajas(zp)
    fs = geo.set_index('canal').fila_salida
    out = {}
    for r in anot.itertuples():
        c, f = int(r.canal), int(r.foto)
        if c not in fs.index or not 1 <= f <= T: continue
        g = cajas[(cajas.canal == c) & (cajas.foto == f)].sort_values('fila_sup')
        cj = [(float(b.fila_sup) + 0.5, float(b.fila_inf) + 0.5, 1.0) for b in g.itertuples() if b.fila_inf > b.fila_sup + 1]
        zona = (max(b for _, b, _ in cj) + ZONA_MANUAL) if cj else float(fs[c]) + 10.0
        out.setdefault(c, {})[f - 1] = (cj, zona)
    return out


def predecir(ruta_img, modelo=None):
    import segmentar_kimogramas as SK
    ruta_img = Path(ruta_img)
    img, geo = abrir(ruta_img)
    base = ruta_img.stem.replace('_canales_BF', '')
    kimo = ruta_img.parent.parent / 'Kimogramas' / f'{base}_kimogramas.npz'
    tiempo = np.load(kimo, allow_pickle=True)['tiempo_min'] if kimo.exists() else np.arange(int(geo.fotos[0]), dtype=float)
    uf = ultima_foto(ruta_img)
    if uf is not None:
        tiempo = tiempo[:uf]
        print(f'{ruta_img.name}: only up to frame {uf} (fotos_validas.csv)')
    malas = fotos_malas(ruta_img.with_name(ruta_img.stem + '_cajas.zip'))
    if len(malas):
        print(f'{ruta_img.name}: {len(malas)} frames marked as defective (skipped)')
    import segmentar_kimogramas as SK
    # 1) per-row probabilities in all channels: with the centre-based detector (its boxes) or with the tree-based one
    cc = cajas_centros_posicion(ruta_img, img, geo) if DETECTOR_CAJAS == 'centros' else None
    conf_resto = None
    if cc is not None:
        # the tracking uses the class of each row according to the network (background, cell, poles). Tested
        # against using its boxes: in LB 03–05 it gives more reasonable complete cycles (90 % vs 76 % with the
        # boxes and 87 % with the tree-based detector) and less spread in rates and durations
        Pc = [cc[k]['P'] for k in range(len(geo))]
        conf_resto = CONF_LINAJE
        print(f'{ruta_img.name}: centre-based detector')
    else:
        if modelo is None: modelo = joblib.load(MODELO)
        clf = modelo['clf']; Pc = []
        for k, (_, g) in enumerate(geo.iterrows()):
            F = caracteristicas_canal(img, geo, k, uf)
            T, L, nf = F.shape
            Pc.append(clf.predict_proba(F.reshape(-1, nf)).reshape(T, L, 4).astype(np.float32))
        print(f'{ruta_img.name}: tree-based detector')
    # frames you corrected by hand: the tracking uses your boxes (those for [8] are still the network's)
    T_ = Pc[0].shape[0]
    man = cajas_manuales(ruta_img, geo, T_) if cc is not None else {}
    cajas_seg = {}
    if man:
        for k, (_, g) in enumerate(geo.iterrows()):
            mc = man.get(int(g.canal))
            if not mc: continue
            cj = list(cc[k]['cajas'][:T_]); Pk = Pc[k].copy(); L = Pk.shape[1]
            for t, (cm, zona) in mc.items():
                malas_c = malas_del_canal(malas, int(g.canal)) if len(malas) else []
                if t + 1 in malas_c: continue
                cj[t] = sorted(cm + [x for x in cj[t] if x[0] >= zona])     # your boxes + the network's below your zone
                z = int(min(max(np.ceil(zona), 0), L))
                Pk[t, :z] = P_de_cajas_centros([cm], L)[0, :z]
            cajas_seg[k] = cj; Pc[k] = Pk
        nf = sum(len(v) for v in man.values())
        print(f'{ruta_img.name}: {nf} channel-frames corrected by hand in {len(man)} channels: the tracking uses your boxes')
    # 2) expected frame-to-frame growth and typical length at birth, from all channels
    T = Pc[0].shape[0]; tiempo = np.asarray(tiempo[:T], float)
    segs = [[s for t in range(0, T, 3) for s in refinar(decodificar(P[t]), P[t]) if s[1] - 0.5 < g.fila_salida - 1]
            for P, (_, g) in zip(Pc, geo.iterrows())]
    tasa, mu, l0 = parametros_posicion([(P[:, :, 2] + P[:, :, 3]).T for P in Pc], segs, geo, tiempo)
    prm = dict(SK.PARAM_DETECTOR, l0=l0)
    # 3) each channel: mother and daughter, boxes, divisions and all cells that can be tracked
    cajas, mhs, divs, revisar, lin_f, lin_c = [], [], [], [], [], []
    args = [(P, g, tiempo, malas_del_canal(malas, int(g.canal)), tasa, prm, True, conf_resto,
             cajas_seg.get(k, cc[k]['cajas'])[:len(tiempo)] if cc is not None else None, mu,
             {t: z for t, (_, z) in man.get(int(g.canal), {}).items()} or None)
            for k, (P, (_, g)) in enumerate(zip(Pc, geo.iterrows()))]
    n_proc = max(1, min(PROCESOS, (os.cpu_count() or 1), len(args)))
    if n_proc > 1:                                       # one channel per process
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_proc) as ex:
            res = list(ex.map(_canal_args, args))
    else:
        res = [_canal_args(a) for a in args]
    for cj, md, dv, rv, lin in res:
        cajas += cj; mhs.append(md); divs += dv; revisar += rv
        if lin is not None:
            lin_f.append(lin[0]); lin_c.append(lin[1])
    cajas = pd.DataFrame(cajas)
    md = pd.concat(mhs, ignore_index=True); ci = SK.ciclos_madre(md)
    dv = pd.DataFrame(divs, columns=['canal', 'foto', 'celula', 'tipo', 'confianza', 'incertidumbre_px', 'revisar'])
    if len(ci):                                          # confidence of the divisions at which each cycle starts and ends
        cm = dv[(dv.celula == 'madre') & (dv.tipo == 'division')].set_index(['canal', 'foto']).confianza
        ci['conf_nace'] = [cm.get((c, s), np.nan) if s == s else np.nan for c, s in zip(ci.canal, ci.slice_nace)]
        ci['conf_divide'] = [cm.get((c, s), np.nan) if s == s else np.nan for c, s in zip(ci.canal, ci.slice_divide)]
    pref = ruta_img.with_suffix('')
    Path(f'{pref}_detector.txt').write_text(''.join(
        f'{r.canal} {r.foto} {r.fila_sup + 0.5:.2f} {r.fila_inf + 0.5:.2f} {r.celula} {int(r.dudosa)}\n' for r in cajas.itertuples()))
    cajas.to_csv(f'{pref}_detector_cajas.csv', index=False)
    md.to_csv(f'{pref}_detector_madre_hija.csv', index=False)
    ci.to_csv(f'{pref}_detector_ciclos.csv', index=False)
    dv.to_csv(f'{pref}_detector_divisiones.csv', index=False)
    if lin_f:
        pd.concat(lin_f, ignore_index=True).round(2).to_csv(f'{pref}_linaje.csv', index=False)
        pd.concat(lin_c, ignore_index=True).round(4).to_csv(f'{pref}_linaje_ciclos.csv', index=False)
    pd.DataFrame(dict(foto=np.arange(1, T + 1), tiempo_min=tiempo, tasa_esperada_por_min=np.round(mu, 5))).to_csv(
        f'{pref}_crecimiento_posicion.csv', index=False)
    revisar = sorted(set(revisar))
    Path(f'{pref}_revisar.txt').write_text(''.join(f'{c} {f} {m}\n' for c, f, m in revisar))
    if cc is not None:                                   # the boxes for [8] and their doubts, from the centre-based detector
        _, _, _, nrev = escribir_propuestas_centros(ruta_img, geo, cc, revisar_previo=revisar)
        revisar = [None] * nrev
    nd = dv[dv.tipo == 'division']
    print(f'{ruta_img.name}: {len(geo)} channels, {len(cajas)} boxes, {int((nd.celula == "madre").sum())} mother divisions '
          f'and {int((nd.celula == "hija").sum())} daughter divisions ({int(nd.revisar.sum())} doubtful); {len(revisar)} frames to review')


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    p = argparse.ArgumentParser(description='Cell detector for straightened channels.')
    p.add_argument('orden', choices=['entrenar', 'validar', 'probar', 'predecir', 'proponer', 'revision', 'figura'])
    p.add_argument('-n', type=int, default=20, help='revision: number of most-doubtful frames proposed per image')
    p.add_argument('archivos', nargs='+', help='entrenar/validar: *_canales_BF_cajas.zip · predecir: *_canales_BF.tif')
    p.add_argument('--modelo', default=str(MODELO))
    p.add_argument('--canal', type=int, help='figura: channel')
    p.add_argument('--fotos', help='figura: frames, e.g. 60-100')
    a = p.parse_args()
    if a.orden == 'entrenar':
        entrenar(a.archivos, a.modelo)
    elif a.orden == 'validar':
        validar(a.archivos)
    elif a.orden == 'proponer':                            # only the boxes for [8] with the centre-based detector
        for f in a.archivos:
            proponer_centros(f)
    elif a.orden == 'revision':                            # frames to review where the tracking is most in doubt
        for f in a.archivos:
            lista_revision(f, a.n)
    elif a.orden == 'figura':
        import linaje as LN
        fotos = None
        if a.fotos:
            i, j = a.fotos.split('-'); fotos = list(range(int(i), int(j) + 1))
        for f in a.archivos:
            print('Saved:', LN.figura(f, a.canal, fotos))
    elif a.orden == 'probar':
        import joblib
        probar(a.archivos, joblib.load(a.modelo))
    else:
        import joblib
        try:
            m = joblib.load(a.modelo)
        except Exception as e:
            sys.exit(f'Cannot open the model {a.modelo} ({e}). Retrain it on this computer.')
        for f in a.archivos:
            predecir(f, m)
