"""
Automatic segmentation of the cells in the bright-field kymographs (kimogramas.py).

Combines a classifier trained on your traces (Trazar_Kimogramas.ijm) with the physics of the
mother machine:

1) Image (machine learning): a pixel classifier (gradient boosting) gives, at each point of
   the kymograph, the probability that there is a boundary between cells. It uses the mean
   kymograph (5 px) and two wider ones of the same channel (minimum and standard deviation
   across the channel width), at several scales, with their derivatives in position and time.

2) Physics (boundary tracking):
   - boundaries do not cross or merge (two distinct cells do not fuse) and only move
     towards the exit: the displacement between slices is 0 at the closed end and does not
     decrease towards the exit (each cell pushes the ones below it); it is computed for each
     pair of slices with dynamic programming and avoids mistaking one boundary for the next
     when cells move by almost their own length;
   - a boundary that is not visible for a few slices still exists (it is joined across the gap);
   - a new boundary only appears through division, near the middle of a long cell;
   - minimum cell length and, in every slice, at least the mother cell.

3) Physics (mother and first daughter): their lower boundaries (B1 and B2) are the most likely
   path (Viterbi over pairs of boundaries) in which, from one slice to the next:
   - both grow exponentially (a cell does not shorten without dividing);
   - or the mother divides: B1 appears around the middle of the mother and her old boundary
     becomes the daughter's (mother + daughter = the grown mother; split ~0.5);
   - or the daughter divides: it drops to ~half and the other half becomes the 3rd cell;
   a boundary only stops being visible if it leaves through the exit, and no visible boundaries
   are left inside the mother or the daughter. Cells further down are given only as a guide.

Usage:
    # train on the kymographs you have traced (saves modelos/bordes_kimo.joblib)
    python segmentar_kimogramas.py entrenar Kimogramas/*_kimo_BF_trazos.zip

    # check quality with cross-validation by channel (does not train the final model)
    python segmentar_kimogramas.py validar Kimogramas/*_kimo_BF_trazos.zip

    # segment new kymographs (the <base>_kimogramas.npz files made by kimogramas.py)
    python segmentar_kimogramas.py predecir Kimogramas/11-02102026_kimogramas.npz

When predicting it saves, next to the kymograph:
    <base>_kimo_BF_auto.zip   boundaries in the Trazar_Kimogramas.ijm format (origen: auto): the pole
                              of the mother, the mother/daughter boundary and the daughter/3rd-cell
                              boundary (with --todos-los-bordes, also those of the cells further down).
                              When you open the kymograph with the macro it offers to load them so
                              you can correct them; the corrected version is saved in <base>_kimo_BF_trazos.zip.
    <base>_auto_celulas.csv   one row per cell and slice (like kimo_trazos.py)
    <base>_auto_madre_hija.csv  per slice: length of the mother and of the first daughter (px),
                              divisions of the mother and of the daughter, and whether they are outside the channel
    <base>_auto_ciclos.csv    one cycle per mother cell: L0, Lc, division time, growth
                              rate and mother/(mother+daughter) split at division
    <base>_auto_revision.png  kymographs with the boundaries found, for review at a glance

Requires: pip install numpy pandas scipy scikit-learn joblib roifile tifffile matplotlib
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELO = Path(__file__).resolve().parent / 'modelos' / 'bordes_kimo.joblib'
VERSION_CARACT = 1                       # change if the features change
ESCALAS = [(1.0, 0.5), (2.0, 1.0), (4.0, 2.0)]   # (sigma position, sigma time) in px / slices


# =========================================================================== data
def cargar_kimogramas(ruta_npz):
    """Reads <base>_kimogramas.npz. If it comes from an old version without BF_min/BF_desv, recomputes
    them from the aligned stack (in the parent folder)."""
    ruta_npz = Path(ruta_npz)
    z = dict(np.load(ruta_npz, allow_pickle=True))
    if 'BF_min' not in z or 'BF_desv' not in z:
        import tifffile
        from kimogramas import kimograma, EXTRA_INF, ANCHO
        ruta_st = ruta_npz.parent.parent / str(z['stack'])
        print(f'  {ruta_npz.name} has no wide kymographs: computing them from {ruta_st.name}')
        st = tifffile.imread(ruta_st)
        can = pd.DataFrame(z['canales'])
        largo = z['BF'].shape[1]
        ancho_aux = max(ANCHO, int(0.25 * float(z['periodo'])) | 1)
        for clave, red in (('BF_min', 'minimo'), ('BF_desv', 'desv')):
            z[clave] = np.stack([kimograma(st, c, ancho_aux, int(z['margen_sup']), EXTRA_INF, largo, red)
                                 for _, c in can.iterrows()])
    z['canales'] = pd.DataFrame(z['canales'])
    return z


def _norm(K):
    k = K / (np.median(K, axis=0, keepdims=True) + 1e-6)          # removes the flicker between slices
    return (k - np.median(k)) / (np.percentile(k, 90) - np.percentile(k, 10) + 1e-6)


def caracteristicas(Kmed, Kmin, Ksd, fila_fondo, fila_salida):
    """Stack (F, L, T) of features of one channel."""
    fs = []
    for K in (_norm(Kmed), _norm(Kmin), _norm(Ksd)):
        for ss, st in ESCALAS:
            g = gaussian_filter(K, (ss, st))
            gs = np.gradient(g, axis=0)
            fs += [g, gs, np.gradient(gs, axis=0), np.gradient(g, axis=1)]
    L, T = Kmed.shape
    filas = np.arange(L)[:, None] * np.ones((1, T))
    fs += [filas - fila_fondo, filas - fila_salida]
    return np.stack(fs).astype(np.float32)


def caracteristicas_canal(z, k):
    """Features of the channel at index k (0 = first channel) of the npz."""
    cc = z['canales'].iloc[k]
    return caracteristicas(z['BF'][k].astype(float), z['BF_min'][k].astype(float),
                           z['BF_desv'][k].astype(float), cc.fila_fondo, cc.fila_salida)


def mapa_bordes(bordes_canal, L, T):
    """Training labels (L, T): 1 within ±1 px of a traced boundary, -1 (ignored) at 2-3 px,
    0 elsewhere; and mask of the traced region (from the closed end to the last traced boundary + 3 px)."""
    Y = np.zeros((L, T), np.int8); M = np.zeros((L, T), bool)
    for s, g in bordes_canal.groupby('slice'):
        t = s - 1
        if t < 0 or t >= T: continue
        f = g.fila.values
        M[:int(min(L, np.ceil(f.max()) + 4)), t] = True
        for fi in f:
            r = int(round(fi))
            Y[max(0, r - 3):r + 4, t] = np.where(Y[max(0, r - 3):r + 4, t] == 1, 1, -1)
            Y[max(0, r - 1):r + 2, t] = 1
    return Y, M


def datos_entrenamiento(rutas_trazos):
    """List of traced channels: dict(base, canal, F, Y, M, P=None)."""
    from kimo_trazos import leer_todo
    datos = []
    for rt in rutas_trazos:
        rt = Path(rt)
        base = rt.name.replace('_kimo_BF_trazos.zip', '')
        z = cargar_kimogramas(rt.with_name(f'{base}_kimogramas.npz'))
        todo = leer_todo(rt)                         # without the discarded parts (key [8] in the macro)
        if todo['ajenos']:
            print(f'  {rt.name}: NOT USED, it has traces from another kymograph {todo["ajenos"]}')
            continue
        if set(todo['origenes']) == {'auto'}:
            print(f'  {rt.name}: NOT USED, it is the uncorrected automatic segmentation')
            continue
        bordes, huecos = todo['bordes'], todo['huecos']
        can = z['canales']
        for c, bc in bordes.groupby('canal'):
            k = int(np.where(can.canal.values == c)[0][0])
            F = caracteristicas_canal(z, k)
            Y, M = mapa_bordes(bc, *F.shape[1:])
            datos.append(dict(base=base, canal=int(c), F=F, Y=Y, M=M, bordes=bc, cc=can.iloc[k],
                              huecos=huecos[huecos.canal == c]))
        desc = todo['descartes']
        print(f'  {rt.name}: {bordes.canal.nunique()} channels traced'
              + (f', {len(desc)} discarded (wholly or in part)' if desc else '')
              + (f', {huecos.groupby(["canal", "hueco"]).ngroups} gaps' if len(huecos) else ''))
    return datos


def _xy(datos, sub=0.3, semilla=0):
    rng = np.random.default_rng(semilla)
    X, y = [], []
    for d in datos:
        sel = d['M'] & (d['Y'] >= 0)
        F, Y = d['F'][:, sel].T, d['Y'][sel]
        keep = (Y == 1) | (rng.random(len(Y)) < sub)
        X.append(F[keep]); y.append(Y[keep])
    return np.concatenate(X), np.concatenate(y)


def _clasificador():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, max_leaf_nodes=31, random_state=0)


def probabilidad(clf, F):
    f, L, T = F.shape
    return clf.predict_proba(F.reshape(f, -1).T)[:, 1].reshape(L, T)


# =========================================================================== physics: boundaries
PARAM = dict(umbral_pico=0.3, sep_min=4, puerta=4.0, max_hueco=4, min_tramo=3, min_score_tramo=0.45,
             long_min=7.0, margen_salida=2.0, tasa_max=0.3, div_r=(0.25, 0.75), tramo_largo=10,
             max_inc=0.5, estaticos=True, propagar=True)
PARAM_MADRE = dict(umbral=0.12, max_cand=9, sigma_g=0.06, caida_tol=0.03, coste_div=4.0, sigma_r=0.12,
                   r_div=(0.3, 0.7), sigma_div=0.15, peso_img=1.0, peso_salto=3.0, long_min=7.0,
                   margen_salida=2.0, coste_fuera=1.5, fuera_del_canal=False, tol_salida=15.0, coste_max=20.0)
# For the detector in the channels (detectar_celulas.py): strictly consistent tracking
# (no cap on the growth cost: two cells cannot merge; in exchange it may skip up to
# 2 bad frames), states scored row by row with the detector's probabilities, and candidates
# also at the predicted position of the boundaries of neighbouring frames (weak boundaries).
PARAM_DETECTOR = dict(PARAM_MADRE, coste_max=None, max_fotos_saltadas=2, coste_foto_saltada=8.0, sigma_g=0.1,
                      peso_img=0.6, max_cand=12, coste_no_visible=4.0, candidatos_vecinos=True,
                      # polar divisions (a filament releases a normal cell from one end) and
                      # fragmentation (it breaks into 3 or more pieces); l0 is estimated for each position
                      divisiones_polares=True, fragmentacion=True, l0=None, factor_filamento=2.5,
                      coste_polar=2.0, sigma_polar=0.4, coste_frag=10.0)
TEMPERATURA = 2.0          # to read the costs as probabilities (confidence of each division)


def picos(p, fila_min, fila_max, umbral, sep_min):
    """Maxima of boundary probability in a profile (subpixel position) and their values."""
    q = gaussian_filter1d(p, 1.0)
    idx, _ = find_peaks(q, height=umbral, distance=sep_min)
    idx = idx[(idx >= max(fila_min, 1)) & (idx <= min(fila_max, len(q) - 2))]
    pos = []
    for i in idx:
        a, b, c = q[i - 1], q[i], q[i + 1]; d = a - 2 * b + c
        pos.append(i + (0.5 * (a - c) / d if d < 0 else 0.0))
    return np.array(pos, float), q[idx]


def desplazamiento(p0, p1, fondo, fila_max, d_min=-1.0, d_max=40.0, paso=0.25, max_inc=0.5, pen=0.02):
    """
    Displacement d(s) of each row between two slices (dynamic programming): ~0 at the closed
    end, non-decreasing towards the exit (cells only grow and push the ones below them)
    and with bounded slope (= local growth rate per slice ≤ max_inc).
    """
    s = np.arange(int(max(0, fondo - 3)), int(fila_max) + 1)
    dk = np.arange(d_min, d_max + 1e-9, paso)
    q = np.interp(s[:, None] + dk[None, :], np.arange(len(p1)), p1, left=0, right=0)
    C = (p0[s][:, None] - q) ** 2
    w = int(round(max_inc / paso)); n, K = C.shape
    D = np.full((n, K), np.inf); B = np.zeros((n, K), np.int16)
    k0 = np.abs(dk) <= 1.0
    D[0, k0] = C[0, k0]
    for i in range(1, n):
        best = np.full(K, np.inf); arg = np.zeros(K, np.int16)
        for inc in range(w + 1):
            prev = np.full(K, np.inf); prev[inc:] = D[i - 1, :K - inc] + pen * inc
            m = prev < best; best[m] = prev[m]; arg[m] = inc
        D[i] = C[i] + best; B[i] = arg
    k = int(np.argmin(D[-1])); ks = [k]
    for i in range(n - 1, 0, -1):
        k -= B[i, k]; ks.append(k)
    return s, dk[np.array(ks[::-1])]


def _alinear(a, pa, b, puerta):
    """Matches the boundaries a (slice t, predicted at pa) to the peaks b (t+1) without changing their order."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), 1e18); B = np.zeros((n + 1, m + 1), np.int8); D[0, 0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            d = D[i, j]
            if d >= 1e18: continue
            if i < n and j < m:
                e = abs(b[j] - pa[i])
                if e <= puerta and b[j] >= a[i] - 1.5:            # a boundary does not move up
                    c = (e / puerta) ** 2
                    if d + c < D[i + 1, j + 1]: D[i + 1, j + 1] = d + c; B[i + 1, j + 1] = 1
            if i < n and d + 1.0 < D[i + 1, j]: D[i + 1, j] = d + 1.0; B[i + 1, j] = 2
            if j < m and d + 1.0 < D[i, j + 1]: D[i, j + 1] = d + 1.0; B[i, j + 1] = 3
    i, j, pares = n, m, []
    while i > 0 or j > 0:
        op = B[i, j]
        if op == 1: pares.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif op == 2: i -= 1
        else: j -= 1
    return pares[::-1]


def seguir_bordes(P, fila_fondo, fila_salida, prm=PARAM, devolver_artefactos=False, info=None):
    """Boundaries tracked over time: {id: {t: fila}} (t starts at 0). If a dict is passed in
    `info`, stores 'lams' in it: typical relative growth from each slice to the next."""
    L, T = P.shape
    fmin, fmax = int(max(1, fila_fondo - 6)), int(min(L - 2, fila_salida - prm['margen_salida']))
    pk = [picos(P[:, t], fmin, fmax, prm['umbral_pico'], prm['sep_min']) for t in range(T)]
    Ps = gaussian_filter(P, (1.5, 0.5))
    flujo = [desplazamiento(Ps[:, t], Ps[:, t + 1], fila_fondo, fmax, max_inc=prm['max_inc']) for t in range(T - 1)]
    # 1) link slice to slice using the physical displacement
    tramos, activos = [], {}
    lam, lams = 0.0, []
    for k, (p, s) in enumerate(zip(*pk[0])):
        activos[k] = len(tramos); tramos.append({0: (p, s)})
    for t in range(T - 1):
        a, _ = pk[t]; b, sb = pk[t + 1]
        dist = np.maximum(a - fila_fondo, 0)
        v = np.maximum(np.interp(a, *flujo[t]), 0) if len(a) else a
        pares = _alinear(a, a + v, b, prm['puerta']) if len(a) and len(b) else []
        rel = [(b[j] - a[i]) / dist[i] for i, j in pares if dist[i] > 10]
        if len(rel) >= 2:                                         # typical rate (for the gaps)
            lam = 0.5 * lam + 0.5 * float(np.clip(np.median(rel), 0, prm['tasa_max']))
        lams.append(lam)
        nuevos, usados = {}, set()
        for i, j in pares:
            if i in activos:
                tid = activos[i]; tramos[tid][t + 1] = (b[j], sb[j]); nuevos[j] = tid; usados.add(j)
        for j in range(len(b)):
            if j not in usados:
                nuevos[j] = len(tramos); tramos.append({t + 1: (b[j], sb[j])})
        activos = nuevos
    lams.append(lam)
    if info is not None:
        info['lams'] = np.array(lams, float)

    def ext(tr):
        ts = sorted(tr); return ts[0], ts[-1]

    def predecir(pos, t0, t1):
        for t in range(t0, t1):
            pos = pos + lams[t] * max(pos - fila_fondo, 0)
        return pos
    # 2a) a boundary does not stop existing because it is not visible for a few slices: join across short gaps
    orden = sorted(range(len(tramos)), key=lambda i: ext(tramos[i])[0])
    vivo = [True] * len(tramos)
    cambiado = True
    while cambiado:
        cambiado = False
        for i in range(len(tramos)):
            if not vivo[i]: continue
            t0, t1 = ext(tramos[i])
            if t1 >= T - 1: continue
            pos = tramos[i][t1][0]; mejor = None
            for j in orden:
                if j == i or not vivo[j]: continue
                s0, _ = ext(tramos[j])
                if t1 < s0 <= t1 + 1 + prm['max_hueco']:
                    pp = predecir(pos, t1, s0)
                    e = abs(tramos[j][s0][0] - pp)
                    if e <= 1.5 * prm['puerta'] and (mejor is None or e < mejor[0]):
                        mejor = (e, j, s0)
            if mejor:
                _, j, s0 = mejor
                p0, p1 = pos, tramos[j][s0][0]
                for t in range(t1 + 1, s0):
                    w = (t - t1) / (s0 - t1); tramos[i][t] = ((1 - w) * p0 + w * p1, 0.0)
                tramos[i].update(tramos[j]); vivo[j] = False; cambiado = True
    tramos = [tr for tr, ok in zip(tramos, vivo) if ok]

    # 2b) noise: short or weak segments
    def score(tr):
        return np.mean([s for _, s in tr.values() if s > 0] or [0])
    tramos = [tr for tr in tramos if len(tr) >= prm['min_tramo'] and score(tr) >= prm['min_score_tramo']]
    # 2c) a boundary only appears mid-experiment through division (near the middle of a
    #     long cell); if it does not look like one and it is short, it is noise
    presentes = {}
    for tid, tr in enumerate(tramos):
        for t, (p, s) in tr.items(): presentes.setdefault(t, []).append((p, tid))
    quitar = set()
    for tid, tr in enumerate(tramos):
        t0, _ = ext(tr)
        if t0 == 0: continue
        p = tr[t0][0]
        otros = sorted(q for q, j in presentes.get(t0, []) if j != tid and j not in quitar)
        arriba = [q for q in otros if q < p]; abajo = [q for q in otros if q > p]
        a = arriba[-1] if arriba else float(fila_fondo)
        division = False
        if abajo:
            r = (p - a) / (abajo[0] - a)
            division = prm['div_r'][0] <= r <= prm['div_r'][1] and abajo[0] - a >= 2 * prm['long_min']
        if not division and len(tr) < prm['tramo_largo']:
            quitar.add(tid)
    tramos = [tr for tid, tr in enumerate(tramos) if tid not in quitar]
    artefactos = []
    # 2d) static lines crossed by a moving boundary: they cannot be boundaries (boundaries do
    #     not cross); they are dirt or marks on the chip
    if prm.get('estaticos', True):
        def vmed(tr):
            ts = sorted(tr); f = np.array([tr[t][0] for t in ts])
            return float(np.median(np.diff(f))) if len(f) > 2 else np.nan
        vel = [vmed(tr) for tr in tramos]
        malos, quietos = set(), {}
        for i, tr in enumerate(tramos):
            if tr is None: continue
            ts = sorted(tr)
            if len(ts) < 5 or not abs(vel[i]) < 0.3: continue
            r = float(np.median([tr[t][0] for t in ts]))
            if r < fila_fondo + 8: continue                       # mother's pole
            quietos[i] = r
            for j, tj in enumerate(tramos):                       # does a moving boundary cross it while it is present?
                if tj is None or j == i or not vel[j] > 1.0 or len(tj) < 6: continue
                com = [t for t in sorted(tj) if t in tr and tj[t][1] > 0 and tr[t][1] > 0]
                d = np.array([tj[t][0] - tr[t][0] for t in com])
                if (d < -3).sum() >= 2 and (d > 3).sum() >= 2:      # they cross: the static one is spurious
                    malos.add(i); break
            # a moving boundary that "breaks off" just above the static line and reappears just
            # below it has crossed it (it is the same boundary: the two pieces are joined)
            unido = True
            while unido:
                unido = False
                for j, tj in enumerate(tramos):
                    if tj is None or j == i or not vel[j] > 1.0: continue
                    te = max(tj); pe = tj[te][0]
                    if not (ts[0] - 2 <= te <= ts[-1] + 2 and r - 15 <= pe < r): continue
                    for k, tk in enumerate(tramos):
                        if tk is None or k in (i, j) or not vel[k] > 1.0: continue
                        t_s = min(tk); pk_ = tk[t_s][0]
                        prev = pe + vel[j] * (t_s - te)
                        if te < t_s <= te + 4 and r < pk_ <= r + 25 and abs(pk_ - prev) <= max(6, 0.5 * vel[j] * (t_s - te)):
                            for t in range(te + 1, t_s):
                                w = (t - te) / (t_s - te); tj[t] = ((1 - w) * pe + w * pk_, 0.0)
                            tj.update(tk); tramos[k] = None; malos.add(i); unido = True
                            break
                    if unido: break
        filas_malas = [quietos[i] for i in malos]                 # the chip mark stays at that row
        if prm.get('propagar', True):
            malos |= {i for i, r in quietos.items() if any(abs(r - q) <= 4 for q in filas_malas)}
        artefactos += [tramos[i] for i in malos if tramos[i] is not None]
        tramos = [tr for i, tr in enumerate(tramos) if i not in malos and tr is not None]
    # 2e) per slice: order, minimum length (the boundary with more support wins) and mother's pole
    por_t = {}
    for tid, tr in enumerate(tramos):
        for t, (p, s) in tr.items(): por_t.setdefault(t, []).append((p, s, tid))
    apoyo = [sum(1 for _, s in tr.values() if s > 0) for tr in tramos]
    final = {}
    for t in range(T):
        lst = sorted(por_t.get(t, []))
        if not lst or lst[0][0] > fila_fondo + 8:
            lst = [(float(fila_fondo + 1), 0.0, -1)] + lst
        res = [lst[0]]
        for p, s, tid in lst[1:]:
            if p - res[-1][0] < prm['long_min']:
                prev = res[-1]
                if prev[2] != -1 and len(res) > 1 and apoyo[tid] > apoyo[prev[2]]:
                    res[-1] = (p, s, tid)
                continue
            res.append((p, s, tid))
        if len(res) == 1:
            res.append((min(res[0][0] + 20.0, fmax), 0.0, -2))
        for p, s, tid in res:
            final.setdefault(tid, {})[t] = p
    if devolver_artefactos:
        return final, artefactos
    return final


# =========================================================================== physics: mother
# =========================================================================== physics: mother and daughter
def _candidatos(q, t, lo, hi, prm):
    idx, _ = find_peaks(q[:, t], height=prm['umbral'])
    idx = idx[(idx > lo) & (idx <= hi)]
    pos = []
    for i in idx:
        a, b, c = q[i - 1, t], q[i, t], q[i + 1, t]; d = a - 2 * b + c
        pos.append(i + (0.5 * (a - c) / d if d < 0 else 0.0))
    pos, sc = np.array(pos, float), q[idx, t]
    if len(pos) > prm['max_cand']:
        k = np.argsort(pos)[:prm['max_cand']]; pos, sc = pos[k], sc[k]
    return pos, sc


def _crec(rho, tasa, prm):
    """Cost of growing log(L'/L) = rho (a cell does not shorten without dividing nor lengthen abruptly).
    With coste_max (old version) an odd slice costs at most that, but this allows two
    cells to "merge"; without a cap, robustness comes from skipping frames (max_fotos_saltadas)."""
    c = (np.where(rho < -prm['caida_tol'], 1e3 * (rho + prm['caida_tol']) ** 2 + 6.0, 0.0)
         + ((rho - tasa) / prm['sigma_g']) ** 2)
    return np.minimum(c, prm['coste_max']) if prm.get('coste_max') else c


def _estados(P, fila_fondo, fila_salida, polo, prm, vacio=None):
    """Per slice: candidate (B1, B2) pairs (nan = outside the channel) and their cost according to the image."""
    L, T = P.shape
    fmax = int(min(L - 2, fila_salida - prm['margen_salida']))
    if prm['fuera_del_canal']: fmax = L - 3
    q = gaussian_filter1d(P, 1.0, axis=0)
    est, em = [], []
    for t in range(T):
        pos, sc = _candidatos(q, t, polo[t] + prm['long_min'], fmax, prm)
        if len(pos) < 3:                                     # bad slice (out of focus...): lower the threshold
            pos, sc = _candidatos(q, t, polo[t] + prm['long_min'], fmax, dict(prm, umbral=prm['umbral'] / 4))
        n = len(pos)
        cum = np.concatenate([[0], np.cumsum(sc)])          # cumulative sum of the peaks above
        e1 = -prm['peso_img'] * np.log(sc + 1e-3)
        S, E = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if pos[j] - pos[i] < prm['long_min']: continue
                S.append((pos[i], pos[j]))
                # peaks skipped inside the mother (before i) and the daughter (between i and j)
                E.append(e1[i] + e1[j] + prm['peso_salto'] * (cum[i] + cum[j] - cum[i + 1]))
            S.append((pos[i], np.nan))                       # daughter extends to the exit
            E.append(e1[i] + prm['coste_fuera'] + prm['peso_salto'] * cum[i] - prm['peso_img'] * np.log(0.05)
                     + prm['peso_salto'] * (cum[n] - cum[i + 1]))
        S.append((np.nan, np.nan))                           # mother extends to the exit
        E.append(2 * prm['coste_fuera'] - prm['peso_img'] * np.log(0.05) + prm['peso_salto'] * cum[n])
        S = np.array(S); E = np.array(E)
        if vacio is not None:
            # occupancy (detector in the channel): there can be no empty space inside the mother
            # or the daughter; the "empty" probability of the rows they cover is added
            cv = np.concatenate([[0], np.cumsum(vacio[:, t])])
            def acum(a, b):
                a = np.clip(np.round(a).astype(int), 0, len(cv) - 1); b = np.clip(np.round(b).astype(int), 0, len(cv) - 1)
                return cv[b] - cv[a]
            p0 = np.full(len(S), polo[t]); b1 = np.where(np.isnan(S[:, 0]), fmax, S[:, 0])
            b2 = np.where(np.isnan(S[:, 1]), fmax, S[:, 1])
            E = E + prm.get('peso_vacio', 1.0) * (acum(p0, b1) + acum(b1, np.maximum(b2, b1)))
        if t == 0:
            E = E + 50.0 * np.isnan(S[:, 0])                # at the start the mother is inside
        est.append(S); em.append(E)
    return est, em, fmax


def _costes_filas(p, eps=1e-4):
    """Per-row costs (-log prob) of the detector classification (background, cell, top pole, bottom pole)."""
    c_bg = -np.log(p[:, 0] + p[:, 2] + p[:, 3] + eps)
    c_in = -np.log(p[:, 1] + eps)
    c_ts = -np.log(p[:, 1] + p[:, 2] + eps)
    c_bs = -np.log(p[:, 1] + p[:, 3] + eps)
    c_top = -np.log(p[:, 2] + eps); c_bot = -np.log(p[:, 3] + eps)
    S_in = np.concatenate([[0], np.cumsum(c_in)]); S_bg = np.concatenate([[0], np.cumsum(c_bg)])
    return c_bg, c_in, c_ts, c_bs, c_top, c_bot, S_in, S_bg


class _Filas:
    """Cost of splitting the rows of a frame into cells and background, with the same model as the
    detector's decoder: inside a cell the rows are "cell", at its ends there are
    poles; a cell cut off by the end of the channel has no bottom pole."""
    def __init__(self, p, F, min_len=5, pen=1.0):
        (self.c_bg, self.c_in, self.c_ts, self.c_bs, self.c_top, self.c_bot, self.S_in, self.S_bg) = _costes_filas(p)
        self.F, self.min_len, self.pen, self.L = int(F), min_len, pen, len(p)
        self.suf = self._sufijo()

    def seg(self, j, i):
        """Cell in rows [j, i) (arrays)."""
        j = np.clip(np.asarray(j), 0, self.L - 3); i = np.clip(np.asarray(i), 3, self.L)
        c = (self.c_ts[j] + self.c_ts[j + 1] + self.c_top[j] + (self.S_in[np.maximum(i - 2, j + 2)] - self.S_in[j + 2])
             + self.c_bot[i - 1] + self.c_bs[i - 1] + self.c_bs[i - 2] + self.pen)
        return np.where(i - j >= self.min_len, c, np.inf)

    def abierta(self, j):
        """Cell from row j to the end of the channel (no bottom pole)."""
        j = np.clip(np.asarray(j), 0, self.F - 3)
        c = self.c_ts[j] + self.c_ts[j + 1] + self.c_top[j] + (self.S_in[self.F] - self.S_in[j + 2]) + self.pen
        return np.where(self.F - j >= self.min_len, c, np.inf)

    def bg(self, a, b):
        a = np.clip(np.asarray(a), 0, self.L); b = np.clip(np.asarray(b), 0, self.L)
        return np.where(b >= a, self.S_bg[b] - self.S_bg[np.minimum(a, b)], np.inf)

    def _sufijo(self):
        """suf[r]: best cost of rows [r, F) (background and cells; the last one may extend out)."""
        F = self.F; suf = np.full(F + 1, np.inf); suf[F] = 0.0
        for r in range(F - 1, -1, -1):
            b = self.c_bg[r] + suf[r + 1]
            k = np.arange(r + self.min_len, F)
            if len(k): b = min(b, float(np.min(self.seg(np.full(len(k), r), k) + suf[k])))
            b = min(b, float(self.abierta(r)))
            suf[r] = b
        return suf


def _estados_filas(P4, Pb, fila_fondo, fila_salida, polo, prm, tasa=0.02):
    """
    Like _estados, but the cost of each state (pole, B1, B2) is that of the image row by row
    according to the detector: mother = cell in [polo, B1), daughter = cell in [B1, B2) and the
    remaining rows up to the exit with the best possible split. A boundary between two cells needs
    the bottom pole of the upper cell and the top pole of the lower one; a static mark inside a
    long cell does not have them.
    """
    T, L, _ = P4.shape
    fmax = int(min(L - 2, fila_salida - prm['margen_salida']))
    if prm['fuera_del_canal']: fmax = L - 3
    q = gaussian_filter1d(Pb, 1.0, axis=0)
    tasas = np.broadcast_to(np.asarray(tasa, float), (T,)) if np.ndim(tasa) else np.full(T, float(tasa))
    picos_ = []
    for t in range(T):
        pos, _ = _candidatos(q, t, polo[t] + prm['long_min'], fmax, dict(prm, max_cand=99))
        if len(pos) < 3:
            pos, _ = _candidatos(q, t, polo[t] + prm['long_min'], fmax, dict(prm, umbral=prm['umbral'] / 4, max_cand=99))
        picos_.append(pos)
    est, em, filas = [], [], []
    for t in range(T):
        # candidates: the peaks of this frame and, if a boundary is visible in the previous or next frame
        # but not in this one (weak boundary), also its position predicted from growth
        pos = list(picos_[t])
        for dt in (-1, 1):
            u = t + dt
            if not 0 <= u < T or not prm.get('candidatos_vecinos', True): continue
            g = np.exp(tasas[min(t, u)] * (t - u))
            for x in picos_[u]:
                y = polo[t] + (x - polo[u]) * g
                if polo[t] + prm['long_min'] < y <= fmax and not any(abs(y - z) < 2.5 for z in pos):
                    pos.append(y)
        pos = np.sort(np.array(pos, float))[:prm['max_cand']]
        fl = _Filas(P4[t], fmax + 1); filas.append(fl)
        n = len(pos); g = np.round(pos).astype(int)
        jp = int(round(polo[t])) + np.array([-1, 0, 1])
        E_ = (g[:, None] + np.array([-1, 0, 1])[None, :])          # end (exclusive) of the upper cell
        S_ = (g[:, None] + np.array([0, 1, 2])[None, :])           # start of the lower one
        # mother [jp, e) + gap [e, s): F1[i, b]
        ME = np.min(fl.seg(jp[None, None, :], E_[:, :, None]), axis=2)                  # (n, 3)
        F1 = np.min(ME[:, :, None] + fl.bg(E_[:, :, None], S_[:, None, :]), axis=1)     # (n, 3) per s
        # after the daughter's lower boundary: gap + rest
        K = np.min(fl.bg(E_[:, :, None], S_[:, None, :]) + fl.suf[np.clip(S_, 0, fmax + 1)][:, None, :], axis=2)  # (n, 3)
        D = fl.seg(S_[:, :, None, None], E_[None, None, :, :])                         # (n, 3, n, 3)
        tot = np.min(F1[:, :, None, None] + D + K[None, None, :, :], axis=(1, 3))       # (n, n)
        S, E = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if pos[j] - pos[i] < prm['long_min']: continue
                S.append((pos[i], pos[j])); E.append(tot[i, j])
            # daughter "outside": it reaches the exit (open cell) or is not visible (bright zone next to the
            # exit: those rows are explained by the best possible split, paying coste_no_visible)
            S.append((pos[i], np.nan))
            E.append(float(np.min(F1[i] + np.minimum(fl.abierta(S_[i]),
                                                     fl.suf[np.clip(S_[i], 0, fmax + 1)] + prm.get('coste_no_visible', 4.0)))))
        S.append((np.nan, np.nan)); E.append(float(np.min(fl.abierta(jp))))
        S = np.array(S); E = np.array(E, float)
        E = prm['peso_img'] * (E - np.min(E[np.isfinite(E)])) if np.isfinite(E).any() else np.zeros(len(E))
        E = np.where(np.isfinite(E), E, 1e3)
        if t == 0:
            E = E + 50.0 * np.isnan(S[:, 0])
        est.append(S); em.append(E)
    _estados_filas.filas = filas                              # to reuse them (cascade)
    return est, em, fmax


def _coste_reparto(q, total, prm):
    """
    Cost of the split of a division (without coste_div). q: fraction of the dividing cell that
    ends up in one of the pieces; total: (grown) length of the dividing cell.
      - symmetric: q ≈ 0.5 (within r_div);
      - polar (only if the cell is long, ≥ factor_filamento · l0): a piece the size of a normal
        cell (l0 = typical length at birth) is released from one end.
    """
    lr = np.log
    with np.errstate(invalid='ignore', divide='ignore'):
        sim = np.where((q >= prm['r_div'][0]) & (q <= prm['r_div'][1]), ((q - 0.5) / prm['sigma_r']) ** 2, np.inf)
        l0 = prm.get('l0')
        if not l0:
            return sim
        trozo = np.minimum(q, 1 - q) * total
        pol = prm.get('coste_polar', 2.0) + (lr(np.maximum(trozo, 1e-3) / l0) / prm.get('sigma_polar', 0.4)) ** 2
        pol = np.where((total >= prm.get('factor_filamento', 2.5) * l0) & (trozo >= prm['long_min'])
                       & (q > 0) & (q < 1), pol, np.inf)
    return np.minimum(sim, pol)


def _transicion(A, Bs, polo_a, polo_b, tasa, fmax, prm):
    """
    Cost (3, nA, nB) of going from state A to Bs with event 0 (nothing), 1 (the mother divides) or
    2 (the daughter divides); `tasa` is the expected (log) growth between the two slices; inf = impossible.
      - nothing: mother and daughter grow (log L'/L ≈ tasa);
      - the mother divides: B1 appears at ~the middle of the mother and B2 is the old B1, displaced
        (mother' + daughter' = grown mother, split r = mother'/(mother'+daughter') ≈ 0.5);
      - the daughter divides: the mother grows and the daughter drops to ~half (the other half becomes the 3rd cell).
    No transition merges two cells: a division is forever.
    """
    lr = np.log
    a1 = (A[:, 0] - polo_a)[:, None]; a2 = (A[:, 1] - A[:, 0])[:, None]
    b1 = (Bs[:, 0] - polo_b)[None, :]; b2 = (Bs[:, 1] - Bs[:, 0])[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        rm = lr(np.maximum(b1, 1) / np.maximum(a1, 1))
        rd = lr(np.maximum(b2, 1) / np.maximum(a2, 1))
        c0 = _crec(rm, tasa, prm) + _crec(rd, tasa, prm)
        tot = b1 + b2
        r = b1 / tot
        g_ = np.exp(tasa)
        if prm.get('divisiones_polares'):
            c1 = (prm['coste_div'] + _crec(lr(np.maximum(tot, 1) / np.maximum(a1, 1)), tasa, prm)
                  + _coste_reparto(r, tot, prm))
            c2 = prm['coste_div'] + _crec(rm, tasa, prm) + _coste_reparto(np.exp(rd - tasa), a2 * g_, prm)
        else:
            c1 = (prm['coste_div'] + _crec(lr(np.maximum(tot, 1) / np.maximum(a1, 1)), tasa, prm)
                  + ((r - 0.5) / prm['sigma_r']) ** 2)
            c1 = np.where((r >= prm['r_div'][0]) & (r <= prm['r_div'][1]), c1, np.inf)
            rdd = rd - tasa
            c2 = prm['coste_div'] + _crec(rm, tasa, prm) + ((rdd - lr(0.5)) / prm['sigma_div']) ** 2
            c2 = np.where((rdd >= lr(prm['r_div'][0])) & (rdd <= lr(prm['r_div'][1])), c2, np.inf)
    # states with parts outside the channel. Physics: a boundary only stops being visible if it
    # leaves through the exit (its predicted position reaches the exit); when the mother divides,
    # her old lower boundary becomes the daughter's, so the daughter can only be "outside" if
    # that boundary is already leaving
    na1, na2 = np.isnan(a1), np.isnan(a2); nb1, nb2 = np.isnan(b1), np.isnan(b2)
    g = np.exp(tasa)
    pa1 = polo_a + np.nan_to_num(a1, nan=1e4) * g             # predicted old B1
    pa2 = polo_a + np.nan_to_num(a1 + a2, nan=1e4) * g        # predicted old B2
    sale1 = pa1 >= fmax - prm['tol_salida']
    sale2 = pa2 >= fmax - prm['tol_salida']
    crm = _crec(rm, tasa, prm)
    en_rango = (rm - tasa >= lr(prm['r_div'][0])) & (rm - tasa <= lr(prm['r_div'][1]))
    divm = np.where(en_rango, prm['coste_div'] + ((rm - tasa - lr(0.5)) / prm['sigma_div']) ** 2, np.inf)
    if prm.get('divisiones_polares'):
        with np.errstate(invalid='ignore', over='ignore'):
            divm = prm['coste_div'] + _coste_reparto(np.exp(rm - tasa), a1 * g, prm)
    nev = 4 if prm.get('fragmentacion') else 3
    C = np.full((nev, len(A), len(Bs)), np.inf)
    def pon(mask, coste, ev):
        coste = np.broadcast_to(coste, C.shape[1:])
        m = mask & (coste < C[ev])
        C[ev] = np.where(m, coste, C[ev])
    ok = ~na1 & ~na2 & ~nb1 & ~nb2
    pon(ok, c0, 0); pon(ok, c1, 1); pon(ok, c2, 2)
    # (a1,a2) -> (b1,·out): the daughter leaves (her boundary reaches the exit), or the mother divides
    # and her old boundary (the new daughter) is leaving
    m = ~na1 & ~na2 & ~nb1 & nb2
    pon(m & sale2, crm, 0); pon(m & sale1, divm, 1)
    # (a1,·out) -> (b1,·out): the mother grows, or divides with the old B1 leaving
    m = ~na1 & na2 & ~nb1 & nb2
    pon(m, crm, 0); pon(m & sale1, divm, 1)
    # (a1,·out) -> (b1,b2): the daughter divides (her boundary appears inside) or the mother does (b2 = old B1)
    m = ~na1 & na2 & ~nb1 & ~nb2
    pon(m, crm + prm['coste_div'], 2); pon(m, c1, 1)
    # mother outside: stays outside, or divides (B1 appears inside)
    pon(na1 & nb1, np.zeros(C.shape[1:]), 0); pon(na1 & ~nb1, np.full(C.shape[1:], prm['coste_div']), 1)
    # the mother stops being visible: her boundary leaves through the exit
    pon(~na1 & nb1 & sale1, np.ones(C.shape[1:]), 0)
    # 3) fragmentation: a long filament breaks into 3 or more pieces between two frames (new B1 and
    #    B2 inside the old mother; her old boundary becomes the 3rd one or further down)
    if nev == 4 and prm.get('l0'):
        with np.errstate(invalid='ignore'):
            p3 = pa1 - Bs[None, :, 1]                              # piece between the new B2 and the old B1
            okf = (~na1 & ~nb1 & ~nb2 & (b1 >= prm['long_min']) & (b2 >= prm['long_min'])
                   & (p3 >= prm['long_min']) & (a1 * g >= prm.get('factor_filamento', 2.5) * prm['l0']))
        pon(okf, np.full(C.shape[1:], prm.get('coste_frag', 10.0)), 3)
    return np.nan_to_num(C, nan=np.inf)


def _rejilla(P, fila_fondo, fila_salida, tasa=0.02, polo=None, prm=PARAM_MADRE, vacio=None, P4=None, malas=None):
    """
    Grid for tracking the mother and daughter: states per slice (_estados) and transitions
    (_transicion) between consecutive slices and, if prm['max_fotos_saltadas'] > 0, also skipping
    1..max_fotos_saltadas slices (bad frame: out of focus, dirt…) with a cost coste_foto_saltada
    per skipped slice. The slices in `malas` (marked by hand as defective) are always skipped, at
    no cost. Returns (est, em, trans, polo, ini, fin): trans[(t0, t1)] = cost (3, n_t0, n_t1);
    ini[t] / fin[t] = cost of starting / ending the path at t (skipping the frames before / after).
    """
    L, T = P.shape
    if polo is None: polo = np.full(T, float(fila_fondo + 1))
    if P4 is not None:                                       # detector probabilities per row
        est, em, fmax = _estados_filas(P4, P, fila_fondo, fila_salida, polo, prm, tasa)
    else:
        est, em, fmax = _estados(P, fila_fondo, fila_salida, polo, prm, vacio)
    tasas = np.broadcast_to(np.asarray(tasa, float), (T,)) if np.ndim(tasa) else np.full(T, float(tasa))
    S = int(prm.get('max_fotos_saltadas', 0)); cs = prm.get('coste_foto_saltada', 0.0)
    mala = np.zeros(T, bool)
    if malas is not None:
        mala[[t for t in malas if 0 <= t < T]] = True
    for t in np.where(mala)[0]:
        em[t] = np.full(len(est[t]), np.inf)
    buenas = np.concatenate([[0], np.cumsum(~mala)])         # buenas[t] = non-defective frames before t
    ini = np.where(buenas[:T] <= S, cs * buenas[:T], np.inf)
    desp = buenas[T] - buenas[1:T + 1]                       # non-defective frames after t
    fin = np.where(desp <= S, cs * desp, np.inf)
    ini[mala] = np.inf; fin[mala] = np.inf
    trans = {}
    for t in range(1, T):
        if mala[t]: continue
        t0, nm = t - 1, 0                                    # nm: good frames skipped
        while t0 >= 0:
            if not mala[t0]:
                C = _transicion(est[t0], est[t], polo[t0], polo[t], float(np.sum(tasas[t0:t])), fmax, prm)
                trans[(t0, t)] = C + cs * nm
                nm += 1
                if nm > S: break
            t0 -= 1
    return est, em, trans, polo, ini, fin


def _lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide='ignore'):
        return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def _bordes_camino(est, cam, polo, tasa):
    """B1, B2 (and pole) per slice along the path; skipped slices are filled with the
    previous state, grown (and flagged)."""
    T = len(est)
    b = np.full((T, 2), np.nan); salt = np.ones(T, bool); polo = np.array(polo, float)
    for t, j in cam: b[t] = est[t][j]; salt[t] = False
    ult = min(t for t, _ in cam) if cam else None          # before the first slice used: backwards
    for t in range(T):
        if not salt[t]: ult = t; continue
        if ult is None: continue
        g = np.exp(tasa * (t - ult))
        polo[t] = polo[ult]
        b[t, 0] = polo[t] + (b[ult, 0] - polo[ult]) * g
        b[t, 1] = b[t, 0] + (b[ult, 1] - b[ult, 0]) * g
    return b, salt, polo


def madre_hija_viterbi(P, fila_fondo, fila_salida, tasa=0.02, polo=None, prm=PARAM_MADRE, vacio=None, temp=None,
                       P4=None, malas=None):
    """
    Most likely path of mother and daughter (see _rejilla and _transicion). Returns a dict with
    per-slice arrays: b1, b2 (rows; nan if outside the channel), polo, saltado (frame that the
    tracking does not use: B1 and B2 are those of the previous slice, grown) and lists of slices
    (from 0) with mother and daughter divisions. With temp (temperature of the costs) it adds the confidence:
      p_div_madre, p_div_hija: probability of division in each slice;
      p_camino: probability of the chosen state in each slice;
      sd_b1, sd_b2: uncertainty (standard deviation, rows) of B1 and B2.
    `malas`: defective slices (from 0), which are always skipped.
    """
    _estados_filas.filas = None
    est, em, trans, polo, ini, fin_ = _rejilla(P, fila_fondo, fila_salida, tasa, polo, prm, vacio, P4, malas)
    T = len(est)
    tasa_m = float(np.mean(tasa))
    # the path may start (and end) by skipping the first (last) frames
    D = [em[0] + ini[0]] + [None] * (T - 1); BK = [None] * T
    for t in range(1, T):
        best = em[t] + ini[t]
        arg = np.full((len(est[t]), 3), -1)                  # (t0, k, event)
        if np.isfinite(ini[t]): arg[:, 0] = -2               # starts here
        for (t0, t1), C3 in trans.items():
            if t1 != t: continue
            ev = np.argmin(C3, axis=0); C = np.min(C3, axis=0)
            tot_ = D[t0][:, None] + C
            k = np.argmin(tot_, axis=0); jj = np.arange(len(k))
            v = tot_[k, jj] + em[t]
            m = v < best
            best = np.where(m, v, best)
            arg[m] = np.stack([np.full(m.sum(), t0), k[m], ev[k[m], jj[m]]], 1)
        D[t], BK[t] = best, arg
    fin = [(D[t].min() + fin_[t], t) for t in range(T) if np.isfinite(fin_[t])]
    t = min(fin)[1]; j = int(np.argmin(D[t])); cam = []; dm, dh, fr = [], [], []
    while True:
        cam.append((t, j))
        if t == 0 or BK[t][j, 0] == -2 or BK[t][j, 0] < 0: break
        t0, k, ev = BK[t][j]
        if ev == 1: dm.append(t)
        elif ev == 2: dh.append(t)
        elif ev == 3: dm.append(t); fr.append(t)
        t, j = int(t0), int(k)
    cam = cam[::-1]
    b, salt, polo_c = _bordes_camino(est, cam, polo, tasa_m)
    out = dict(b1=b[:, 0], b2=b[:, 1], div_madre=sorted(dm), div_hija=sorted(dh), frag_madre=sorted(fr),
               polo=polo_c, saltado=salt)
    if P4 is not None and getattr(_estados_filas, 'filas', None) is not None:
        out['filas'] = _estados_filas.filas                   # per-row costs of each frame (for the cascade)
    if temp:
        pm, ph, gamma = _posterior(est, em, trans, temp, ini, fin_)
        jc = dict(cam)
        out.update(p_div_madre=pm, p_div_hija=ph,
                   p_camino=np.array([gamma[t][jc[t]] if t in jc else 0.0 for t in range(T)]),
                   p_saltado=np.clip(1 - np.array([g.sum() for g in gamma]), 0, 1))
        for n, col in ((0, 'sd_b1'), (1, 'sd_b2')):
            sd = np.full(T, np.nan)
            for t in range(T):
                x = est[t][:, n]; m = ~np.isnan(x); w = gamma[t][m]
                if w.sum() > 0.5:
                    mu = np.sum(w * x[m]) / w.sum(); sd[t] = np.sqrt(np.sum(w * (x[m] - mu) ** 2) / w.sum())
            out[col] = sd
    return out


def _posterior(est, em, trans, temp, ini, fin):
    """
    Forward-backward over the same grid as the Viterbi: costs are read as
    -log(probability) * temp. Returns the probability that the mother (p_div_madre) or the daughter
    (p_div_hija) divides in each slice (the division is assigned to the first slice in which it is seen)
    and the probability of each state (gamma; its sum is < 1 if the frame may have been skipped).
    """
    T = len(est)
    with np.errstate(invalid='ignore'):
        la = [None] * T
        for t in range(T):
            terms = [(-em[t] - ini[t]) / temp]
            for (t0, t1), C3 in trans.items():
                if t1 == t:
                    terms.append(_lse(la[t0][None, :, None] - C3 / temp, axis=(0, 1)) - em[t] / temp)
            la[t] = _lse(np.stack(terms), axis=0)
        lb = [None] * T
        for t in range(T - 1, -1, -1):
            terms = [np.full(len(est[t]), -fin[t] / temp)]
            for (t0, t1), C3 in trans.items():
                if t0 == t:
                    terms.append(_lse(-C3 / temp + (lb[t1] - em[t1] / temp)[None, None, :], axis=(0, 2)))
            lb[t] = _lse(np.stack(terms), axis=0)
        lz = _lse(np.concatenate([la[t] - fin[t] / temp for t in range(T)]), axis=0)
    pm, ph = np.zeros(T), np.zeros(T)
    gamma = [np.exp(la[t] + lb[t] - lz) for t in range(T)]
    for (t0, t1), C3 in trans.items():
        xi = np.exp(la[t0][None, :, None] - C3 / temp + (lb[t1] - em[t1] / temp)[None, None, :] - lz)
        pm[t1] += xi[1].sum() + (xi[3].sum() if len(xi) > 3 else 0.0); ph[t1] += xi[2].sum()
    return pm, ph, gamma


def tasa_posicion(lams_canales, tiempo, ventana=5):
    """
    Expected (log) growth from each slice to the next for a whole position, from the rate
    measured by seguir_bordes in each channel ('lams', relative per slice): converted to a rate per
    minute using the actual time of each frame, median across channels and rolling median over time.
    This way the tracking knows how much a cell should grow between two frames even if the interval
    or the medium changes. Returns (rate per slice, rate per minute).
    """
    tiempo = np.asarray(tiempo, float); T = len(tiempo)
    dt = np.diff(tiempo); dt = np.append(dt, np.median(dt) if len(dt) else 1.0); dt = np.maximum(dt, 1e-3)
    M = []
    for l in lams_canales:
        l = np.asarray(l, float)[:T].copy()
        nz = np.nonzero(l)[0]
        if len(nz): l[:nz[0]] = np.nan                      # before the first measurement: no data
        else: continue
        M.append(np.log1p(np.clip(l, -0.5, None)) / dt)
    if not M:
        mu = np.full(T, 0.02 / max(np.median(dt), 1e-3))
    else:
        mu = np.nanmedian(np.array(M), axis=0)
        mu = pd.Series(mu).rolling(ventana, center=True, min_periods=1).median().interpolate(limit_direction='both').to_numpy()
        mu = np.where(np.isfinite(mu), mu, np.nanmedian(mu) if np.isfinite(mu).any() else 0.005)
    mu = np.clip(mu, 0, None)
    return mu * dt, mu


def _tasa(final, fila_fondo):
    rel = []
    for tid, d in final.items():
        if tid < 0: continue
        ts = sorted(d)
        for a, b in zip(ts, ts[1:]):
            if b == a + 1 and d[a] - fila_fondo > 15:
                rel.append(np.log((d[b] - fila_fondo) / (d[a] - fila_fondo)))
    rel = np.array(rel); rel = rel[(rel > -0.02) & (rel < 0.3)]
    return float(np.median(rel)) if len(rel) > 10 else 0.02


def segmentar_canal(P, fila_fondo, fila_salida, prm=PARAM, prm_madre=PARAM_MADRE, solo_madre_hija=False, vacio=None,
                    temp=None, P4=None, P_bordes=None, malas=None, tasa=None, polo_min=None):
    """
    Segmentation of one channel. P: boundary map (rows x slices) used to track the boundaries and
    the mother's pole. Optional: P_bordes (map for the mother and daughter candidates, if different),
    P4 (detector probabilities per frame and row: states are scored row by row) and temp
    (adds the confidence: probability of each division and uncertainty of the boundaries). malas:
    defective slices (from 0), which the mother-daughter tracking skips. tasa: expected (log)
    growth from each slice to the next (array); by default, a fixed one estimated in the channel itself.
    polo_min: per slice, row where the first cell starts if there is an empty gap above it (nan = no).
    Returns (bordes, mh):
      bordes: {id: {t: fila}} with the mother's pole, B1 (mother/daughter), B2 (daughter/3rd cell) and
              the remaining boundaries tracked further down;
      mh:     dict with per-slice arrays polo, b1, b2 (nan = outside the channel), saltado and the lists
              of slices (from 0) with divisions of the mother (div_madre) and of the daughter (div_hija);
              with temp also p_div_madre, p_div_hija, p_camino, p_saltado, sd_b1, sd_b2.
    """
    final, artefactos = seguir_bordes(P, fila_fondo, fila_salida, prm, devolver_artefactos=True)
    tasa = _tasa(final, fila_fondo) if tasa is None else tasa
    T = P.shape[1]
    Pm = (P if P_bordes is None else P_bordes).copy()   # the mother cannot end on an artefact
    for tr in artefactos:
        for t, (f, _) in tr.items():
            Pm[max(0, int(round(f)) - 3):int(round(f)) + 4, t] = 0
    por_t = {}
    for tid, d in final.items():
        for t, f in d.items(): por_t.setdefault(t, []).append((f, tid))
    polo = np.full(T, float(fila_fondo + 1))
    for t in range(T):
        lst = sorted(por_t.get(t, []))
        if lst and lst[0][0] < fila_fondo + 8: polo[t] = lst[0][0]
    if polo_min is not None:                        # empty gap above: the mother starts further down
        pm = np.asarray(polo_min, float)
        polo = np.where(np.isfinite(pm), np.maximum(polo, pm), polo)
    mh = madre_hija_viterbi(Pm, fila_fondo, fila_salida, tasa, polo, prm_madre, vacio, temp, P4, malas)
    mh['tasa'] = np.broadcast_to(np.asarray(tasa, float), (T,)).copy() if np.ndim(tasa) else np.full(T, float(tasa))
    b1, b2 = mh['b1'], mh['b2']
    dm, dh = set(mh['div_madre']), set(mh['div_hija'])
    # identities: B1 changes at each division of the mother and her old boundary becomes B2;
    # B2 changes when the daughter divides
    nuevo = {'polo': {}}
    sig = 0
    id1 = id2 = None
    for t in range(T):
        nuevo['polo'][t] = float(polo[t])
        if np.isnan(b1[t]):
            id1 = id2 = None
        else:
            if t in dm or id1 is None:
                viejo = id1
                sig += 1; id1 = f'm{sig}'
                id2 = viejo if viejo is not None and not np.isnan(b2[t]) else None
            if np.isnan(b2[t]):
                id2 = None
            elif id2 is None or t in dh:
                sig += 1; id2 = f'm{sig}'
            nuevo.setdefault(id1, {})[t] = float(b1[t])
            if id2 is not None:
                nuevo.setdefault(id2, {})[t] = float(b2[t])
        fondo_mh = b2[t] if not np.isnan(b2[t]) else (b1[t] if not np.isnan(b1[t]) else np.inf)
        for f, tid in ([] if solo_madre_hija else por_t.get(t, [])):   # remaining boundaries, below the daughter
            if tid >= 0 and f > fondo_mh + prm['long_min']:
                nuevo.setdefault(tid, {})[t] = float(f)
    ids = sorted(nuevo, key=lambda k: (min(nuevo[k]), nuevo[k][min(nuevo[k])]))
    return {n + 1: nuevo[k] for n, k in enumerate(ids)}, mh


def tabla_madre_hija(mh, canal, tiempo):
    """One row per slice: lengths of the mother and of the first daughter, and divisions."""
    T = len(mh['b1'])
    df = pd.DataFrame(dict(canal=canal, slice=np.arange(1, T + 1), tiempo_min=tiempo[:T],
                           polo=mh['polo'], borde_madre=mh['b1'], borde_hija=mh['b2']))
    df['L_madre_px'] = df.borde_madre - df.polo
    df['L_hija_px'] = df.borde_hija - df.borde_madre
    df['madre_fuera'] = df.borde_madre.isna()        # the mother extends beyond the exit
    df['hija_fuera'] = df.borde_hija.isna()
    df['div_madre'] = df.slice.isin([t + 1 for t in mh['div_madre']])
    df['div_hija'] = df.slice.isin([t + 1 for t in mh['div_hija']])
    if 'saltado' in mh:
        df['foto_saltada'] = mh['saltado']           # bad frame: mother and daughter = those of the previous frame, grown
    for k in ('p_div_madre', 'p_div_hija', 'p_camino', 'sd_b1', 'sd_b2'):
        if k in mh: df[k] = np.round(mh[k], 3)
    return df


def ciclos_madre(mh_tabla, min_slices=3):
    """
    One cycle per mother cell: from the division at which it is born to the next one. Also the
    initial segment (born before the experiment) and the final one (does not reach division), with
    completo=False. Columns: L0 (at birth) and Lc (just before dividing) in px, division
    time, growth rate (exponential fit of L(t), per minute), split at the division at which it
    is born (mother / (mother + daughter)) and a warning if the cycle is suspiciously short.
    """
    filas = []
    for c, g in mh_tabla.groupby('canal'):
        g = g.sort_values('slice').reset_index(drop=True)
        cortes = [0] + list(g.index[g.div_madre]) + [len(g)]
        for k in range(len(cortes) - 1):
            i0, i1 = cortes[k], cortes[k + 1] - 1
            if i1 < i0: continue
            nace, divide = k > 0, k + 1 < len(cortes) - 1
            tramo = g.loc[i0:i1]
            v = tramo.dropna(subset=['L_madre_px']); v = v[v.L_madre_px > 0]
            tasa = np.polyfit(v.tiempo_min, np.log(v.L_madre_px), 1)[0] if len(v) >= 3 else np.nan
            r0, r1 = g.loc[i0], g.loc[i1]
            filas.append(dict(
                canal=c, slice_nace=int(r0.slice) if nace else np.nan, slice_divide=int(g.loc[i1 + 1].slice) if divide else np.nan,
                t_div_min=(g.loc[i1 + 1].tiempo_min - r0.tiempo_min) if nace and divide else np.nan,
                L0_px=r0.L_madre_px if nace else np.nan, Lc_px=r1.L_madre_px if divide else np.nan,
                tasa_por_min=tasa, reparto=(r0.L_madre_px / (r0.L_madre_px + r0.L_hija_px)) if nace else np.nan,
                slices=len(tramo), completo=nace and divide,
                aviso='ciclo muy corto: ¿división falsa?' if len(tramo) < min_slices and nace and divide else
                      ('madre fuera del canal' if tramo.madre_fuera.any() else '')))
    return pd.DataFrame(filas)


def a_tabla(bordes, canal):
    return pd.DataFrame([dict(canal=canal, borde=b, slice=t + 1, fila=f)
                         for b, d in bordes.items() for t, f in sorted(d.items())])


# =========================================================================== comparison with traces
def comparar(pred, manual, tol=4.0):
    """Boundaries slice by slice within the traced region: precision, recall, F1 and fraction of slices
    with the exact number of cells."""
    tp = fp = fn = 0; dn = []
    for (c, s), gm in manual.groupby(['canal', 'slice']):
        fm = np.sort(gm.fila.values)
        gp = pred[(pred.canal == c) & (pred.slice == s)]
        fp_ = np.sort(gp.fila.values); fp_ = fp_[fp_ <= fm.max() + 3]
        if len(fp_) and len(fm):
            C = np.abs(fm[:, None] - fp_[None, :]); r, k = linear_sum_assignment(C)
            ok = C[r, k] <= tol; tp += ok.sum(); fn += len(fm) - ok.sum(); fp += len(fp_) - ok.sum()
        else:
            fn += len(fm); fp += len(fp_)
        dn.append(len(fp_) - len(fm))
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9), n_celulas_exacto=float(np.mean(np.array(dn) == 0)))


def comparar_madre_hija(auto, manual, canales, tol=3.0, huecos=None):
    """
    Automatic mother and daughter (tabla_madre_hija) versus your traces: in each slice your first
    boundary is the mother's pole, the second B1 and the third B2. Fraction of slices with the length
    within `tol` px and within 10 %, and mother divisions (your B1 drops below 70 %).
    """
    fondo = canales.set_index('canal').fila_fondo
    hh = {}
    if huecos is not None and len(huecos):
        hh = {k: g.fila.to_numpy() for k, g in huecos.groupby(['canal', 'slice'])}
    filas = []
    for (c, s), g in manual.groupby(['canal', 'slice']):
        f = np.sort(g.fila.values)
        if f[0] > fondo[c] + 8 or len(f) < 2: continue
        if (c, s) in hh and np.any(hh[(c, s)] < (f[2] if len(f) > 2 else f[1])):
            continue                                 # there is a gap next to the mother or the daughter
        filas.append(dict(canal=c, slice=s, Lm=f[1] - f[0], Ld=f[2] - f[1] if len(f) > 2 else np.nan))
    V = pd.DataFrame(filas).merge(auto, on=['canal', 'slice'])
    em = (V.Lm - V.L_madre_px).abs().fillna(np.inf); vd = V[V.Ld.notna()]
    ed = (vd.Ld - vd.L_hija_px).abs().fillna(np.inf)
    dm = []
    for c, g in V.sort_values('slice').groupby('canal'):
        lm, s = g.Lm.values, g.slice.values
        dm += [(c, s[k]) for k in range(1, len(lm)) if s[k] == s[k - 1] + 1 and lm[k] < 0.7 * lm[k - 1] and s[k] > 3]
    traz = set(zip(V.canal, V.slice))
    da = [(c, s) for c, s in zip(auto.canal[auto.div_madre], auto.slice[auto.div_madre]) if (c, s) in traz]
    ok = sum(1 for c, s in dm if any(c == c2 and abs(s - s2) <= 2 for c2, s2 in da))
    return dict(madre_menos_3px=float(np.mean(em < tol)), madre_menos_10pc=float(np.mean(em / V.Lm < 0.1)),
                hija_menos_3px=float(np.mean(ed < tol)) if len(vd) else np.nan,
                hija_menos_10pc=float(np.mean(ed / vd.Ld < 0.1)) if len(vd) else np.nan,
                div_madre_tuyas=len(dm), div_madre_auto=len(da), div_madre_coinciden=ok)


# =========================================================================== output
def _simplificar(t, f, tol=0.5):
    """Douglas-Peucker: fewer vertices in the polyline (easier to edit in Fiji)."""
    keep = np.zeros(len(t), bool); keep[[0, -1]] = True
    pila = [(0, len(t) - 1)]
    while pila:
        i, j = pila.pop()
        if j <= i + 1: continue
        tt, ff = t[i:j + 1], f[i:j + 1]
        lin = ff[0] + (ff[-1] - ff[0]) * (tt - tt[0]) / (tt[-1] - tt[0])
        k = int(np.argmax(np.abs(ff - lin)))
        if abs(ff[k] - lin[k]) > tol:
            keep[i + k] = True; pila += [(i, i + k), (i + k, j)]
    return t[keep], f[keep]


def guardar_rois(tabla, ruta_zip, kimograma, indice_canal):
    """Boundaries -> zip of ROIs in the Trazar_Kimogramas.ijm format (K05_b003, polyline)."""
    import zipfile
    import roifile
    colores = ['#ffff00', '#00ffff', '#ff00ff', '#00ff00', '#ff8000', '#4080ff', '#ffffff', '#ff80c0', '#80ff80', '#c080ff']
    ahora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with zipfile.ZipFile(ruta_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for (c, b), g in tabla.groupby(['canal', 'borde']):
            g = g.sort_values('slice')
            t = g.slice.to_numpy(float) - 1; f = g.fila.to_numpy(float)
            if len(t) == 1: t, f = np.array([t[0], t[0] + 0.5]), np.array([f[0], f[0]])
            t, f = _simplificar(t, f)
            pts = np.c_[t + 0.5, f + 0.5].astype(np.float32)          # pixel centre -> ImageJ
            roi = roifile.ImagejRoi.frompoints(pts)
            roi.roitype = roifile.ROI_TYPE.POLYLINE
            nombre = f'K{int(c):02d}_b{int(b):03d}'
            roi.name = nombre
            roi.position = int(indice_canal[c])
            roi.stroke_color = bytes([255]) + bytes.fromhex(colores[(int(b) - 1) % len(colores)].lstrip('#'))
            roi.props = (f'canal: {int(c)}\ntipo: borde\nslice_ini: {int(g.slice.min())}\nslice_fin: {int(g.slice.max())}\n'
                         f'kimograma: {kimograma}\norigen: auto\ntrazado_el: {ahora}\n')
            zf.writestr(nombre + '.roi', roi.tobytes())


def figura_revision(z, tablas, ruta, titulo, mh=None, por_fila=8):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from kimogramas import normalizar
    can = z['canales']; n = len(can)
    filas = int(np.ceil(n / por_fila))
    fig, axs = plt.subplots(filas, por_fila, figsize=(2.3 * por_fila, 4.2 * filas), squeeze=False)
    for ax in axs.flat: ax.axis('off')
    for k, (_, cc) in enumerate(can.iterrows()):
        ax = axs.flat[k]
        ax.imshow(normalizar(z['BF'][k]), cmap='gray', aspect='auto')
        tb = tablas.get(int(cc.canal))
        if tb is not None:
            for _, g in tb.groupby('borde'):
                ax.plot(g.slice - 1, g.fila, lw=0.5, color='#ffd54a', alpha=0.7)
        if mh is not None:
            g = mh[mh.canal == int(cc.canal)]
            ax.plot(g.slice - 1, g.borde_madre, color='#ff3b30', lw=1.0)
            ax.plot(g.slice - 1, g.borde_hija, color='#00e5ff', lw=1.0)
            for s in g.slice[g.div_madre]:
                ax.axvline(s - 1, color='#ff3b30', lw=0.4, alpha=0.6)
        ax.set_title(f'channel {int(cc.canal):02d}', fontsize=8)
    fig.suptitle(titulo, fontsize=10); plt.tight_layout()
    fig.savefig(ruta, dpi=70); plt.close(fig)


# =========================================================================== commands
def entrenar(rutas_trazos, ruta_modelo=MODELO):
    import joblib
    t0 = time.time()
    datos = datos_entrenamiento(rutas_trazos)
    X, y = _xy(datos)
    print(f'Training on {len(datos)} channels, {len(y)} pixels ({int(y.sum())} boundary)…')
    clf = _clasificador().fit(X, y)
    ruta_modelo = Path(ruta_modelo); ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(fecha=datetime.now().strftime('%Y-%m-%d %H:%M'), version_caracteristicas=VERSION_CARACT,
                kimogramas=[Path(r).name for r in rutas_trazos], canales=len(datos), pixeles=int(len(y)))
    joblib.dump(dict(clf=clf, meta=meta), ruta_modelo)
    ruta_modelo.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f'Saved: {ruta_modelo} ({time.time() - t0:.0f} s)')


def validar(rutas_trazos, pliegues=5):
    """Cross-validation by channel: each channel is segmented with a model that has not seen it."""
    datos = datos_entrenamiento(rutas_trazos)
    rng = np.random.default_rng(1)
    grupo = rng.integers(pliegues, size=len(datos))
    for k in range(pliegues):
        tr = [d for d, g in zip(datos, grupo) if g != k]
        X, y = _xy(tr); clf = _clasificador().fit(X, y)
        for d, g in zip(datos, grupo):
            if g == k: d['P'] = probabilidad(clf, d['F'])
        print(f'  fold {k + 1}/{pliegues} done')
    filas = []
    for base in dict.fromkeys(d['base'] for d in datos):
        ds = [d for d in datos if d['base'] == base]
        preds, mhs = [], []
        for d in ds:
            b, mh = segmentar_canal(d['P'], d['cc'].fila_fondo, d['cc'].fila_salida)
            preds.append(a_tabla(b, d['canal'])); mhs.append(tabla_madre_hija(mh, d['canal'], np.arange(d['P'].shape[1])))
        pred = pd.concat(preds); man = pd.concat([d['bordes'] for d in ds])
        r = comparar(pred, man)
        r.update(comparar_madre_hija(pd.concat(mhs), man, pd.DataFrame([d['cc'] for d in ds]),
                                     huecos=pd.concat([d['huecos'] for d in ds])))
        filas.append(dict(kimograma=base, canales=len(ds), **r))
    res = pd.DataFrame(filas).set_index('kimograma')
    with pd.option_context('display.width', 200):
        print(res.round(2).to_string())
    return res


def predecir(ruta_npz, ruta_modelo=MODELO, modelo=None, solo_madre_hija=True):
    import joblib
    from kimo_trazos import celulas
    ruta_npz = Path(ruta_npz)
    base = ruta_npz.name.replace('_kimogramas.npz', '')
    if modelo is None:
        modelo = joblib.load(ruta_modelo)
    clf = modelo['clf']
    z = cargar_kimogramas(ruta_npz)
    can = z['canales']
    tiempo = z['tiempo_min']
    tablas, mhs = {}, []
    for k, (_, cc) in enumerate(can.iterrows()):
        P = probabilidad(clf, caracteristicas_canal(z, k))
        bordes, mh = segmentar_canal(P, cc.fila_fondo, cc.fila_salida, solo_madre_hija=solo_madre_hija)
        tablas[int(cc.canal)] = a_tabla(bordes, int(cc.canal))
        mhs.append(tabla_madre_hija(mh, int(cc.canal), tiempo))
    todo = pd.concat(tablas.values(), ignore_index=True)
    carpeta = ruta_npz.parent
    indice = {int(c): k + 1 for k, c in enumerate(can.canal)}
    guardar_rois(todo, carpeta / f'{base}_kimo_BF_auto.zip', f'{base}_kimo_BF.tif', indice)
    cels = celulas(todo, can)
    cels['tiempo_min'] = tiempo[cels.slice.to_numpy() - 1]
    if 'canal_usuario' in can: cels['canal_usuario'] = cels.canal.map(can.set_index('canal').canal_usuario)
    cels.to_csv(carpeta / f'{base}_auto_celulas.csv', index=False)
    md = pd.concat(mhs, ignore_index=True)
    ci = ciclos_madre(md)
    if 'canal_usuario' in can:
        md['canal_usuario'] = md.canal.map(can.set_index('canal').canal_usuario)
        if len(ci): ci['canal_usuario'] = ci.canal.map(can.set_index('canal').canal_usuario)
    md.to_csv(carpeta / f'{base}_auto_madre_hija.csv', index=False)
    ci.to_csv(carpeta / f'{base}_auto_ciclos.csv', index=False)
    figura_revision(z, tablas, carpeta / f'{base}_auto_revision.png', f'{base} – automatic segmentation (red: mother, cyan: daughter)', md)
    print(f'{base}: {len(can)} channels, {int(md.div_madre.sum())} mother divisions, '
          f'{int(ci.completo.sum()) if len(ci) else 0} complete cycles -> {base}_kimo_BF_auto.zip, _auto_madre_hija.csv, _auto_ciclos.csv')
    return todo, cels, md, ci


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    p = argparse.ArgumentParser(description='Automatic segmentation of kymographs (machine learning + physics).')
    p.add_argument('orden', choices=['entrenar', 'validar', 'predecir'])
    p.add_argument('archivos', nargs='+', help='entrenar/validar: *_kimo_BF_trazos.zip · predecir: *_kimogramas.npz')
    p.add_argument('--modelo', default=str(MODELO), help=f'default: {MODELO}')
    p.add_argument('--todos-los-bordes', action='store_true',
                   help='predecir: also save the boundaries of the cells below the daughter (by default only pole, mother and daughter)')
    a = p.parse_args()
    if a.orden == 'entrenar':
        entrenar(a.archivos, a.modelo)
    elif a.orden == 'validar':
        validar(a.archivos)
    else:
        import joblib
        try:
            m = joblib.load(a.modelo)
        except Exception as e:                      # e.g. a different scikit-learn version
            sys.exit(f'Cannot open the model {a.modelo} ({e}).\n'
                     'Retrain it on this computer (it takes a few seconds):\n'
                     '    python segmentar_kimogramas.py entrenar <carpeta>/Kimogramas/*_kimo_BF_trazos.zip')
        print(f"Model: {a.modelo} (trained on {m['meta']['fecha']} with {', '.join(m['meta']['kimogramas'])})")
        for f in a.archivos:
            predecir(f, modelo=m, solo_madre_hija=not a.todos_los_bordes)
