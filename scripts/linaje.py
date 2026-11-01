"""
Lineage in the channel: every edge that can be tracked, not just mother and daughter.

With the centre-based detector (the network) the lineage comes from seguir_cajas_global (further down): its
boxes are tracked by solving the whole movie at once (or frame by frame with seguir_cajas, if that fails),
allowing gaps between cells. What follows below is the cascade (tree-based detector).

1) Mother and daughter (B1, B2) with the joint tracking from segmentar_kimogramas.py.
2) Cascade downwards: edge B3 is tracked knowing where B1 and B2 are in each frame; then B4
   knowing B1..B3, etc. Geometry rules: when a cell above divides, a new edge appears
   above it and all the edges below move down one place (the edge that was in place
   j-1 becomes j). In each frame, edge j can: continue (its cell grows as expected),
   be a new one because its cell has divided (symmetric or polar split), leave through the exit or
   stop being visible (bright zone next to the exit).
3) From the edges in each frame and where new edges appear, the lineage is reconstructed: each
   cell with its mother, when it is born, when it divides or leaves, and its length in each frame.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import segmentar_kimogramas as SK


def _candidatos_todos(q, polo, fmax, prm):
    """Peaks of the edge map in each frame (all of them, sorted)."""
    T = q.shape[1]; out = []
    for t in range(T):
        pos, _ = SK._candidatos(q, t, polo[t] + prm['long_min'], fmax, dict(prm, max_cand=99))
        if len(pos) < 3:
            pos, _ = SK._candidatos(q, t, polo[t] + prm['long_min'], fmax, dict(prm, umbral=prm['umbral'] / 4, max_cand=99))
        out.append(np.sort(pos))
    return out


def _emision(fl, u, ys, fmax, prm):
    """Image cost of cell j between edge u (known) and each candidate y; nan = the cell
    extends to the exit or nothing below it is visible."""
    s = int(round(u)) + np.array([0, 1, 2])
    E = []
    for y in ys:
        if np.isnan(y):
            E.append(float(np.min(np.minimum(fl.abierta(s), fl.suf[np.clip(s, 0, fmax + 1)] + prm.get('coste_no_visible', 4.0)))))
            continue
        e = int(round(y)) + np.array([-1, 0, 1]); s2 = int(round(y)) + np.array([0, 1, 2])
        c = (fl.seg(s[:, None, None], e[None, :, None]) + fl.bg(e[None, :, None], s2[None, None, :])
             + fl.suf[np.clip(s2, 0, fmax + 1)][None, None, :])
        E.append(float(np.min(c)))
    E = np.array(E)
    fin = np.isfinite(E)
    E = prm['peso_img'] * (E - (E[fin].min() if fin.any() else 0)); E[~fin] = 1e3
    return E


def etapa(j, U, nuevos, polo, filas, picos, tasas, fmax, prm, saltar, temp=None):
    """
    Tracks edge j (j ≥ 3) given the edges above it. U[i] = position of edge i (1..j-1) in each frame
    (nan = outside); nuevos[t] = places of the edges that appear in frame t (divisions).
    Returns (y: position of edge j per frame, eventos: frames in which cell j divides,
    p_div: division probability of cell j in each frame, sd: uncertainty of the edge).
    """
    T = len(polo); lr = np.log; tol = prm['tol_salida']
    up = U[j - 1]
    est, em = [], []
    for t in range(T):
        u = up[t]
        if np.isnan(u):
            ys = np.array([np.nan])
        else:
            c = [y for y in picos[t] if y > u + prm['long_min']]
            for dt in (-1, 1):                              # weak edges: position predicted from the neighbouring frame
                v = t + dt
                if 0 <= v < T and not np.isnan(up[v]):
                    g = np.exp(tasas[min(t, v)] * (t - v))
                    for x in picos[v]:
                        if x > up[v] + prm['long_min']:
                            yy = polo[t] + (x - polo[v]) * g
                            if u + prm['long_min'] < yy <= fmax and not any(abs(yy - z) < 2.5 for z in c):
                                c.append(yy)
            c = sorted(c)[:prm.get('max_cand_cascada', 10)]
            ys = np.array(c + [np.nan])
        est.append(ys)
        em.append(np.full(len(ys), np.inf) if saltar[t] else (_emision(filas[t], u, ys, fmax, prm) if not np.isnan(u) else np.zeros(1)))
    # number of new edges above place j in each frame
    m_ar = np.array([sum(1 for i in nuevos[t] if i < j) for t in range(T)])

    def trans(t0, t1):
        A, B = est[t0], est[t1]
        tasa = float(np.sum(tasas[t0:t1])); g = np.exp(tasa)
        m = int(m_ar[t0 + 1:t1 + 1].sum())
        nuevos_t1 = set(nuevos[t1])
        u0, u1 = up[t0], up[t1]
        C = np.full((2, len(A), len(B)), np.inf)
        b_len = B - u1                                      # new length of cell j
        with np.errstate(invalid='ignore', divide='ignore'):
            if m == 0:
                # the same cell j as before: it grows (or divides at the same time)
                a_len = A - u0; old_vis = ~np.isnan(A)
                rho = lr(np.maximum(b_len[None, :], 1) / np.maximum(a_len[:, None], 1))
                c0 = SK._crec(rho, tasa, prm)
                q = b_len[None, :] / (a_len[:, None] * g); tot = a_len[:, None] * g
                pred = polo[t1] + (A - polo[t0]) * g
            else:
                # m edges have appeared above: edge j is the old edge j-m, which moves with
                # the expansion of everything above it
                ib = j - m
                if ib < 1:
                    return C
                ub = U[ib][t0]; old_vis = np.full(len(A), not np.isnan(ub))
                pred = np.full(len(A), polo[t1] + (ub - polo[t0]) * g)
                if (j - 1) in nuevos_t1 or ib - 1 < 1 and False:
                    # its upper edge is new: cell j is the lower part of the old cell j-m
                    rho = lr(np.maximum(B - polo[t1], 1) / max(ub - polo[t0], 1))[None, :]
                else:
                    ua = U[ib - 1][t0] if ib - 1 >= 1 else polo[t0]
                    rho = lr(np.maximum(b_len, 1) / max(ub - ua, 1))[None, :]
                c0 = np.broadcast_to(SK._crec(rho, tasa, prm), (len(A), len(B)))
                tot = (pred[:, None] - u1); q = b_len[None, :] / tot
        c1 = prm['coste_div'] + SK._coste_reparto(q, tot, prm)
        vis_a = old_vis[:, None]; vis_b = ~np.isnan(B)[None, :]
        okb = vis_b & (not np.isnan(u1))
        C[0] = np.where(vis_a & okb, c0, C[0])
        C[1] = np.where(vis_a & okb, c1, C[1])
        sale = (pred >= fmax - tol)[:, None]
        no_vis = prm.get('coste_deja_de_verse', 3.0)
        # edge j leaves or stops being visible
        C[0] = np.where(vis_a & ~vis_b, np.where(sale, 0.0, no_vis), C[0])
        # cell j was not visible (or was leaving) and stays that way / an edge appears
        nva = ~old_vis[:, None]
        C[0] = np.where(nva & ~vis_b, 0.0, C[0])
        C[0] = np.where(nva & vis_b, no_vis, C[0])
        C[1] = np.where(nva & vis_b, prm['coste_div'] + no_vis, C[1])
        return np.nan_to_num(C, nan=np.inf)

    S = int(prm.get('max_fotos_saltadas', 0)); cs = prm.get('coste_foto_saltada', 0.0)
    buenas = np.concatenate([[0], np.cumsum(~saltar)])
    ini = np.where(buenas[:T] <= S, cs * buenas[:T], np.inf); ini[saltar] = np.inf
    desp = buenas[T] - buenas[1:T + 1]; fin = np.where(desp <= S, cs * desp, np.inf); fin[saltar] = np.inf
    trs = {}
    for t in range(1, T):
        if saltar[t]: continue
        t0, nm = t - 1, 0
        while t0 >= 0:
            if not saltar[t0]:
                trs[(t0, t)] = trans(t0, t) + cs * nm
                nm += 1
                if nm > S: break
            t0 -= 1
    # Viterbi
    D = [None] * T; BK = [None] * T
    for t in range(T):
        best = em[t] + ini[t]; arg = np.full((len(est[t]), 3), -1); arg[np.isfinite(ini[t]) & np.ones(len(est[t]), bool), 0] = -2
        for (t0, t1), C3 in trs.items():
            if t1 != t: continue
            ev = np.argmin(C3, axis=0); C = np.min(C3, axis=0)
            tot = D[t0][:, None] + C; k = np.argmin(tot, axis=0); jj = np.arange(len(k))
            v = tot[k, jj] + em[t]; msk = v < best
            best = np.where(msk, v, best); arg[msk] = np.stack([np.full(msk.sum(), t0), k[msk], ev[k[msk], jj[msk]]], 1)
        D[t], BK[t] = best, arg
    finales = [(D[t].min() + fin[t], t) for t in range(T) if np.isfinite(fin[t]) and D[t] is not None]
    t = min(finales)[1]; i = int(np.argmin(D[t])); cam = []; evs = []
    while True:
        cam.append((t, i))
        if BK[t][i, 0] < 0: break
        t0, k, ev = BK[t][i]
        if ev == 1: evs.append(t)
        t, i = int(t0), int(k)
    y = np.full(T, np.nan); usado = np.zeros(T, bool)
    for t, i in cam: y[t] = est[t][i]; usado[t] = True
    # skipped frames: the edge from the previous used frame, grown
    ult = None
    for t in range(T):
        if usado[t]: ult = t; continue
        if ult is not None and not np.isnan(y[ult]):
            y[t] = polo[t] + (y[ult] - polo[ult]) * np.exp(np.sum(tasas[ult:t]))
    out = dict(y=y, eventos=sorted(evs))
    if temp:
        pm, gamma = _posterior2(est, em, trs, temp, ini, fin)
        sd = np.full(T, np.nan)
        for t in range(T):
            x = est[t]; mk = ~np.isnan(x); w = gamma[t][mk]
            if w.sum() > 0.5:
                mu = np.sum(w * x[mk]) / w.sum(); sd[t] = np.sqrt(np.sum(w * (x[mk] - mu) ** 2) / w.sum())
        out.update(p_div=pm, sd=sd)
    return out


def _posterior2(est, em, trs, temp, ini, fin):
    T = len(est); lse = SK._lse
    with np.errstate(invalid='ignore'):
        la = [None] * T
        for t in range(T):
            terms = [(-em[t] - ini[t]) / temp]
            for (t0, t1), C in trs.items():
                if t1 == t: terms.append(lse(la[t0][None, :, None] - C / temp, axis=(0, 1)) - em[t] / temp)
            la[t] = lse(np.stack(terms), axis=0)
        lb = [None] * T
        for t in range(T - 1, -1, -1):
            terms = [np.full(len(est[t]), -fin[t] / temp)]
            for (t0, t1), C in trs.items():
                if t0 == t: terms.append(lse(-C / temp + (lb[t1] - em[t1] / temp)[None, None, :], axis=(0, 2)))
            lb[t] = lse(np.stack(terms), axis=0)
        lz = lse(np.concatenate([la[t] - fin[t] / temp for t in range(T)]), axis=0)
        pm = np.zeros(T)
        for (t0, t1), C in trs.items():
            pm[t1] += np.exp(la[t0][:, None] - C[1] / temp + (lb[t1] - em[t1] / temp)[None, :] - lz).sum()
        gamma = [np.exp(la[t] + lb[t] - lz) for t in range(T)]
    return pm, gamma


def cascada(P4, mh, fila_salida, prm, max_bordes=20, temp=None):
    """
    All edges: B1 and B2 from the mother/daughter tracking (mh) and, further down, one by one.
    Returns a dict with 'B' (list of per-frame arrays: B[1], B[2], …), 'nuevos' (per frame, the
    places of the edges that appear), 'tipo_madre' (fragmentation) and the confidence per edge.
    """
    T, L, _ = P4.shape
    fmax = int(min(L - 2, fila_salida - prm['margen_salida']))
    polo = np.asarray(mh['polo'], float); tasas = np.asarray(mh['tasa'], float)
    from detectar_celulas import mapa_bordes
    q = gaussian_filter1d(mapa_bordes(P4), 1.0, axis=0)
    filas = mh.get('filas') or [SK._Filas(P4[t], fmax + 1) for t in range(T)]
    picos = _candidatos_todos(q, polo, fmax, prm)
    saltar = np.asarray(mh['saltado'], bool)
    U = {1: np.asarray(mh['b1'], float), 2: np.asarray(mh['b2'], float)}
    nuevos = [[] for _ in range(T)]
    frag = set(mh.get('frag_madre', []))
    for t in mh['div_madre']:
        nuevos[t] += [1, 2] if t in frag else [1]
    for t in mh['div_hija']:
        nuevos[t].append(2)
    conf = {1: dict(p_div=mh.get('p_div_madre'), sd=mh.get('sd_b1')), 2: dict(p_div=mh.get('p_div_hija'), sd=mh.get('sd_b2'))}
    for j in range(3, max_bordes + 1):
        if np.all(np.isnan(U[j - 1])): break
        r = etapa(j, U, nuevos, polo, filas, picos, tasas, fmax, prm, saltar, temp)
        if np.all(np.isnan(r['y'])): break
        U[j] = r['y']
        for t in r['eventos']: nuevos[t].append(j)
        conf[j] = dict(p_div=r.get('p_div'), sd=r.get('sd'))
    return dict(B=U, nuevos=[sorted(n) for n in nuevos], polo=polo, conf=conf, saltado=saltar)


def linaje(casc, P4, fila_salida, tiempo=None):
    """
    Cells from the edges. Each physical edge gets an identifier when it appears; a
    cell is the stretch between two consecutive edges (or between the pole and B1). When a new
    edge appears inside a cell, that cell has divided in two.
    Returns (por_foto, celulas): por_foto, channel-independent (foto, celula, madre_id, orden,
    fila_sup, fila_inf, longitud_px, sale, sd_px), and celulas (one row per cell).
    """
    from detectar_celulas import _pico
    B = casc['B']; polo = casc['polo']; T = len(polo); K = max(B)
    fmax = fila_salida
    sig = [1]                                                 # next edge identifier
    def nuevo_id():
        sig[0] += 1; return sig[0] - 1
    ids_prev = None; pos_prev = None
    celulas = {}                                              # cid -> dict
    viva = {}                                                 # (borde_sup, borde_inf_al_nacer) -> cid
    sig_c = [1]
    filas = []
    cel_prev = {}                                             # borde_sup -> cid (the cell whose upper edge it is)
    for t in range(T):
        pos = [B[k][t] for k in range(1, K + 1)]
        pos = [p for p in pos if not np.isnan(p)]
        # cut at the first nan (nothing further down is tracked)
        vis = []
        for k in range(1, K + 1):
            if np.isnan(B[k][t]): break
            vis.append(B[k][t])
        nv = set(casc['nuevos'][t]) if t > 0 else set()
        if ids_prev is None:
            ids = [nuevo_id() for _ in vis]; nacidos = set()
        else:
            ids = []; it = iter(ids_prev); nacidos = set()
            for k in range(1, len(vis) + 1):
                if k in nv:
                    b = nuevo_id(); ids.append(b); nacidos.add(b)
                else:
                    b = next(it, None)
                    if b is None:
                        b = nuevo_id()                         # appears (starts being visible)
                    ids.append(b)
        # cells in this frame: (upper edge, lower edge); 0 = pole; None = extends to the exit
        sup_ids = [0] + ids
        inf_ids = ids + [None]
        sup_pos = [polo[t]] + vis
        inf_pos = vis + [np.nan]
        cel_t = {}
        for o, (a, b, ya, yb) in enumerate(zip(sup_ids, inf_ids, sup_pos, inf_pos)):
            # same cell as before: same upper edge and no new edge has appeared inside it
            cid = cel_prev.get(a)
            divid = (b in nacidos) or (a in nacidos)
            if cid is None or divid:
                madre = None
                if a in nacidos:                               # its upper edge is new: the mother is the cell that was above it before
                    madre = cel_prev.get(sup_ids[o - 1]) if o >= 1 else None
                    # the cell above it before the division
                    k_up = o - 1
                    while k_up >= 0 and sup_ids[k_up] in nacidos: k_up -= 1
                    madre = cel_prev.get(sup_ids[k_up]) if k_up >= 0 else None
                elif b in nacidos:
                    madre = cel_prev.get(a)
                cid = sig_c[0]; sig_c[0] += 1
                celulas[cid] = dict(celula=cid, madre_id=madre, nace=t if (madre is not None) else np.nan,
                                    primera=t, ultima=t, final='fin')
                if madre is not None and madre in celulas:
                    celulas[madre]['final'] = 'division'; celulas[madre]['divide'] = t
            celulas[cid]['ultima'] = t
            cel_t[a] = cid
            # box: poles refined with the detector
            p = P4[t]
            sup = _pico(p[:, 2], ya - 3, ya + 3) if a == 0 else _pico(p[:, 2], ya - 1, ya + 3)
            if np.isnan(sup): sup = ya
            if b is None:
                inf = fmax + 2.0; sale = True
            else:
                inf = _pico(p[:, 3], yb - 4, yb) + 1
                if np.isnan(inf) or inf - sup < 4: inf = yb
                sale = bool(inf - 0.5 >= fila_salida - 1)
            filas.append(dict(foto=t + 1, celula=cid, orden=o + 1, fila_sup=sup - 0.5, fila_inf=inf - 0.5,
                              longitud_px=(np.nan if sale else inf - sup), sale_del_canal=sale,
                              saltada=bool(casc['saltado'][t])))
        # cells that were there and no longer are: they leave (their upper edge is no longer visible)
        for a, cid in cel_prev.items():
            if cid not in cel_t.values() and celulas[cid]['final'] == 'fin':
                celulas[cid]['final'] = 'sale'
        cel_prev = cel_t; ids_prev = ids
    por_foto = pd.DataFrame(filas)
    cel = pd.DataFrame(list(celulas.values()))
    return por_foto, cel


# =========================================================================== tracking with gaps
"""
Tracking of the boxes in a channel allowing gaps between cells (ordered alignment).

In the mother machine cells never overtake each other: the top-to-bottom order is preserved. Matching
the cells of one frame with the boxes of the next one is therefore an alignment of two sequences, and it
is solved exactly with dynamic programming. Moves (each with a cost):
  - continue (1 -> 1): the expected length is L·exp(α·Δt) (α specific to each cell, updated as it
    goes) and the expected position follows from the growth of the cells above it that touch
    it; a cell with a gap above it does not move (only if they catch up with it);
  - divide (1 -> 2, or 1 -> 3 for filaments): consecutive boxes adding up to the expected length;
  - exit: the cell reaches the exit;
  - missing: the detector does not see it in this frame; it is still expected for up to MAX_FALTAS frames
    (this way t is matched with t+2) and, if it does not come back, it «disappears» (dies, lyses or is lost);
  - appear: a box that does not come from any cell (at the start of the movie, or an error).
"""

PRM_HUECOS = dict(
    sig_pos=3.0,         # px: uncertainty of the position of the upper pole
    sig_pos_k=0.15,      # + this fraction of the expected displacement
    sig_libre=5.0,       # + this if the cell is not pushed from the closed end (loose block)
    sig_logL=0.07,       # relative uncertainty of the length
    sig_Lpx=2.0,         # + absolute uncertainty of the length (px)
    contacto=4.0,        # px: two cells touch if the gap is smaller than this
    c_div=3.0,           # cost of a division
    c_div3=9.0,          # extra cost for each additional piece (fragmentation of a filament)
    sig_reparto=0.12,    # expected split 0.5 ± this (polar splits are allowed, with a cost)
    c_joven=6.0,         # extra cost if it «divides» before reaching 1.3 times its length at birth
    max_hueco_div=6.0,   # px: maximum gap between freshly divided daughters
    c_falta=5.0,         # cost of not seeing a cell (detector failure, or it dies)
    c_nueva=12.0,        # cost of a box that does not come from any cell
    peso_conf=0.0,       # the cost of a new box is multiplied by (1 - weight + weight·confidence)
    c_salida=0.5,        # cost of leaving when it is expected to reach the exit
    max_faltas=3,        # consecutive frames without seeing it before it is considered gone
    tope=12.0,           # cap on each cost term (robustness)
    tope_pos=8.0,        # cap on the position term
    sig_baja=8.0,        # extra px if the cell has moved down more than expected (dragging; it cannot move up)
    c_une=2.0,           # two newborn sisters that become one box again: the division was false
    fotos_une=3,         # … if they were born at most this many frames ago
    c_div_media=1.5,     # extra for a division in which the detector sees only one of the daughters
    z_dudosa=3.0,        # a box whose length deviates from the expected one by more than this (in SDs) is not measured
    max_salto_min=20.0,  # min without (bad) frames beyond which tracking starts again (it cannot be continued)
    alfa0=0.02,          # 1/min, if there is no rate for the position
    alfa_min=-0.02, alfa_max=0.06, ema=0.5,
)


def _cuad(z, tope):
    return min(0.5 * z * z, tope)


def seguir_cajas(cajas, tiempo, fila_fondo, fila_salida, mu=None, malas=(), prm=None):
    """
    cajas: one list per frame of [(fila_sup, fila_inf, confianza)] (those of a bad frame are ignored).
    tiempo: minutes of each frame. mu: expected rate of the position per frame (1/min), optional.
    Returns (F, C) like linaje.linaje: F per cell and frame, C per cell (madre_id, nace, final…).
    """
    p = dict(PRM_HUECOS, **(prm or {}))
    T = len(cajas); tiempo = np.asarray(tiempo, float); fs = float(fila_salida)
    mala = np.zeros(T, bool); mala[[f - 1 for f in malas if 1 <= f <= T]] = True
    cel = {}                     # id -> dict(madre, nace, final, divide, obs=[(t,a,b,conf,longitud_fiable)])
    vivas = []                   # active cells: dict(id, a, b, L, alfa, faltas, L0, trunc)
    sig = [1]

    def nueva(a, b, c, t, madre=None, alfa=None):
        i = sig[0]; sig[0] += 1
        cel[i] = dict(madre=madre, nace=t if madre is not None else np.nan, final='fin', divide=np.nan,
                      primera=t, ultima=t, obs=[(t, a, b, c, True)], L0=b - a)
        tr = b >= fs - 1.5
        return dict(id=i, a=a, b=b, L=b - a, alfa=alfa if alfa is not None else _alfa0(t), faltas=0, L0=b - a,
                    trunc=tr, t_obs=t, L_obs=b - a)

    def _alfa0(t):
        if mu is not None and t < len(mu) and np.isfinite(mu[t]): return float(mu[t])
        return p['alfa0']

    ok = np.where(~mala)[0]
    dt_tipico = float(np.median(np.diff(tiempo[ok]))) if len(ok) > 2 else 5.0
    t_prev = None
    for t in range(T):
        if mala[t]: continue
        bx = sorted([(max(float(a), fila_fondo - 1.0), float(b), float(c)) for a, b, c in cajas[t] if a < fs - 1 and b > a + 1])
        bx = [x for x in bx if x[1] > x[0] + 1]
        salto = t_prev is not None and tiempo[t] - tiempo[t_prev] > max(p['max_salto_min'], 3 * dt_tipico)
        if salto:
            for v in vivas:
                if cel[v['id']]['final'] == 'fin': cel[v['id']]['final'] = 'perdida'
        if t_prev is None or salto:
            vivas = [nueva(a, b, c, t) for a, b, c in bx]
            t_prev = t; continue
        dt = tiempo[t] - tiempo[t_prev]
        # ---- prediction: cells grow and push the ones touching them from below
        vivas.sort(key=lambda v: v['a'])
        n = len(vivas); ap = np.zeros(n); Lp = np.zeros(n); libre = np.zeros(n, bool); sh = np.zeros(n)
        for i, v in enumerate(vivas):
            Lp[i] = v['L'] * np.exp(v['alfa'] * dt)
            if i == 0:
                ap[i] = v['a']; libre[i] = v['a'] > fila_fondo + 6
            else:
                gap = v['a'] - vivas[i - 1]['b']; bp = ap[i - 1] + Lp[i - 1]
                if gap <= p['contacto']:
                    ap[i] = bp + max(gap, 0.0); libre[i] = libre[i - 1]
                else:
                    ap[i] = max(v['a'], bp + 1.0); libre[i] = ap[i] <= v['a'] + 0.5 or libre[i - 1]
            sh[i] = ap[i] - v['a']
        # ---- costs
        m = len(bx)

        def c_seguir(i, j, k):
            v = vivas[i]; a = bx[j][0]; b = bx[j + k - 1][1]
            for q in range(j, j + k - 1):
                if bx[q + 1][0] - bx[q][1] > p['max_hueco_div']: return np.inf
            s = p['sig_pos'] + p['sig_pos_k'] * abs(sh[i]) + (p['sig_libre'] if libre[i] else 0.0)
            d = a - ap[i]
            if d > 0: s += p['sig_baja']                         # it can be dragged down (not up)
            c = _cuad(d / s, p['tope_pos'])
            Lo = b - a; sL = np.hypot(p['sig_logL'], p['sig_Lpx'] / max(Lp[i], 1.0))
            z = np.log(max(Lo, 1.0) / Lp[i]) / sL
            if b >= fs - 1.5: z = max(z, 0.0)   # cut off by the exit
            c += _cuad(z, p['tope'])
            if k >= 2:
                Ls = [bx[q][1] - bx[q][0] for q in range(j, j + k)]
                c += p['c_div'] + p['c_div3'] * (k - 2)
                if k == 2 and b < fs - 1.5:
                    r = Ls[0] / sum(Ls); c += min(_cuad((r - 0.5) / p['sig_reparto'], 4.0), 4.0)
                if Lp[i] < 1.3 * v['L0'] and not v['trunc']: c += p['c_joven']
            return c

        def c_falta(i):
            if ap[i] >= fs - 2: return 0.0                       # already out
            if ap[i] + Lp[i] >= fs - 1.5: return p['c_salida']  # reaching the exit
            return p['c_falta']

        def c_media(i, j, arriba):
            # it divides but only one daughter is visible (arriba=True: the upper one); the other one is left as «missing»
            v = vivas[i]; a, b, _ = bx[j]; Lo = b - a
            if b >= fs - 1.5 or Lp[i] < 1.3 * v['L0'] or Lp[i] + ap[i] >= fs - 1.5: return np.inf
            r = Lo / Lp[i]
            if not 0.25 <= r <= 0.75: return np.inf
            s = p['sig_pos'] + p['sig_pos_k'] * abs(sh[i]) + (p['sig_libre'] if libre[i] else 0.0)
            d = (a - ap[i]) if arriba else (b - ap[i] - Lp[i])
            if d > 0: s += p['sig_baja']
            return (p['c_div'] + p['c_div_media'] + p['c_falta'] * 0.5 + _cuad(d / s, p['tope_pos'])
                    + min(_cuad((r - 0.5) / p['sig_reparto'], 4.0), 4.0))

        def hermanas(i):
            u, w = cel[vivas[i]['id']], cel[vivas[i + 1]['id']]
            return (u['madre'] is not None and u['madre'] == w['madre'] and u['nace'] == w['nace']
                    and t - u['nace'] <= p['fotos_une'])

        def c_une(i, j):
            a, b, _ = bx[j]; L2 = Lp[i] + Lp[i + 1] + max(vivas[i + 1]['a'] - vivas[i]['b'], 0.0)
            s = p['sig_pos'] + p['sig_pos_k'] * abs(sh[i]) + (p['sig_libre'] if libre[i] else 0.0)
            d = a - ap[i]
            if d > 0: s += p['sig_baja']
            sL = np.hypot(p['sig_logL'], p['sig_Lpx'] / max(L2, 1.0)); z = np.log(max(b - a, 1.0) / L2) / sL
            if b >= fs - 1.5: z = max(z, 0.0)
            return p['c_une'] + _cuad(d / s, p['tope_pos']) + _cuad(z, p['tope'])

        INF = np.inf
        D = np.full((n + 1, m + 1), INF); D[0, 0] = 0.0; B = {}
        for i in range(n + 1):
            for j in range(m + 1):
                d0 = D[i, j]
                if not np.isfinite(d0): continue
                if i < n:
                    c = d0 + c_falta(i)
                    if c < D[i + 1, j]: D[i + 1, j] = c; B[(i + 1, j)] = ('falta', i, j, 0)
                if j < m:
                    c = d0 + p['c_nueva'] * (1 - p['peso_conf'] + p['peso_conf'] * bx[j][2])
                    if c < D[i, j + 1]: D[i, j + 1] = c; B[(i, j + 1)] = ('nueva', i, j, 0)
                if i < n:
                    for k in (1, 2, 3):
                        if j + k > m: break
                        c = d0 + c_seguir(i, j, k)
                        if c < D[i + 1, j + k]: D[i + 1, j + k] = c; B[(i + 1, j + k)] = ('sigue', i, j, k)
                if i < n and j < m:
                    for arriba in (True, False):
                        c = d0 + c_media(i, j, arriba)
                        if c < D[i + 1, j + 1]: D[i + 1, j + 1] = c; B[(i + 1, j + 1)] = ('media', i, j, int(arriba))
                if i + 1 < n and j < m and hermanas(i):
                    c = d0 + c_une(i, j)
                    if c < D[i + 2, j + 1]: D[i + 2, j + 1] = c; B[(i + 2, j + 1)] = ('une', i, j, 0)
        ops = []; ij = (n, m)
        while ij != (0, 0):
            op = B[ij]; ops.append(op); i, j = op[1], op[2]; ij = (i, j)
        ops.reverse()
        # ---- update
        nuevas = []
        for tipo, i, j, k in ops:
            if tipo == 'nueva':
                a, b, c = bx[j]; nuevas.append(nueva(a, b, c, t))
            elif tipo == 'une':                                  # the division was false: the mother comes back
                h1, h2 = vivas[i]['id'], vivas[i + 1]['id']; mid = cel[h1]['madre']; M = cel[mid]
                o1 = {o[0]: o for o in cel[h1]['obs']}; o2 = {o[0]: o for o in cel[h2]['obs']}
                for tt in sorted(set(o1) | set(o2)):
                    q = [o for o in (o1.get(tt), o2.get(tt)) if o is not None]
                    M['obs'].append((tt, min(o[1] for o in q), max(o[2] for o in q), min(o[3] for o in q), all(o[4] for o in q)))
                del cel[h1], cel[h2]
                a, b, c = bx[j]; M['obs'].append((t, a, b, c, True)); M.update(final='fin', divide=np.nan, ultima=t)
                tr = b >= fs - 1.5
                nuevas.append(dict(id=mid, a=a, b=b, L=max(b - a, Lp[i] + Lp[i + 1]) if tr else b - a, alfa=vivas[i]['alfa'],
                                   faltas=0, L0=M['L0'], trunc=tr, t_obs=t, L_obs=(b - a) if not tr else vivas[i]['L_obs']))
            elif tipo == 'falta':
                v = vivas[i]; v['faltas'] += 1
                v['a'] = ap[i]; v['b'] = ap[i] + Lp[i]; v['L'] = Lp[i]
                fuera = ap[i] + Lp[i] >= fs - 1.5
                if fuera or v['faltas'] > p['max_faltas']:
                    cel[v['id']]['final'] = 'sale' if (fuera or v['trunc']) else 'desaparece'
                else:
                    nuevas.append(v)
            elif tipo == 'sigue' and k == 1:
                v = vivas[i]; a, b, c = bx[j]; tr = b >= fs - 1.5
                sL = np.hypot(p['sig_logL'], p['sig_Lpx'] / max(Lp[i], 1.0))
                if not tr and abs(np.log(max(b - a, 1.0) / Lp[i])) > p['z_dudosa'] * sL:
                    # the box does not match the expected length (the network sees only a piece, or two cells
                    # together): same cell, but that length is not used
                    cel[v['id']]['obs'].append((t, a, b, c, False)); cel[v['id']]['ultima'] = t
                    v.update(a=a, b=a + Lp[i], L=Lp[i], faltas=0)
                    nuevas.append(v); continue
                cel[v['id']]['obs'].append((t, a, b, c, True)); cel[v['id']]['ultima'] = t
                if not tr and not v['trunc'] and b - a > 2:
                    al = np.log((b - a) / v['L_obs']) / max(tiempo[t] - tiempo[v['t_obs']], 1e-6)
                    v['alfa'] = float(np.clip(p['ema'] * v['alfa'] + (1 - p['ema']) * al, p['alfa_min'], p['alfa_max']))
                v['L'] = max(b - a, Lp[i]) if tr else b - a
                v.update(a=a, b=b, faltas=0, trunc=tr, t_obs=t)
                if not tr: v['L_obs'] = b - a
                nuevas.append(v)
            elif tipo == 'media':                                  # division with one daughter not seen
                v = vivas[i]; cel[v['id']]['final'] = 'division'; cel[v['id']]['divide'] = t
                a, b, c = bx[j]; h = nueva(a, b, c, t, madre=v['id'], alfa=v['alfa'])
                Lg = max(Lp[i] - (b - a), 2.0)
                if k:   # the upper one is visible; the lower one goes below it
                    ga, gb = b + 1.0, b + 1.0 + Lg
                else:
                    ga, gb = a - 1.0 - Lg, a - 1.0
                gid = sig[0]; sig[0] += 1
                cel[gid] = dict(madre=v['id'], nace=t, final='fin', divide=np.nan, primera=t, ultima=t, obs=[], L0=Lg)
                g = dict(id=gid, a=ga, b=gb, L=Lg, alfa=v['alfa'], faltas=1, L0=Lg, trunc=False, t_obs=t, L_obs=Lg)
                nuevas += [h, g] if k else [g, h]
            else:                                                  # division
                v = vivas[i]; cel[v['id']]['final'] = 'division'; cel[v['id']]['divide'] = t
                for q in range(j, j + k):
                    a, b, c = bx[q]; nuevas.append(nueva(a, b, c, t, madre=v['id'], alfa=v['alfa']))
        vivas = nuevas; t_prev = t
    # ---- tables
    filas = []
    madres = {d['madre'] for d in cel.values()}
    for i in [i for i, d in cel.items() if not d['obs'] and i not in madres]: del cel[i]   # daughters that were never seen
    for i, d in cel.items():
        for t, a, b, c, fiable in d['obs']:
            sale = bool(b >= fs - 1.5)
            filas.append(dict(foto=t + 1, celula=i, fila_sup=round(a, 2), fila_inf=round(b, 2),
                              longitud_px=np.nan if (sale or not fiable) else round(b - a, 2), sale_del_canal=sale, saltada=False,
                              fila_inf_visible=round(min(b, fs - 1), 2), confianza=round(c, 3), visible=True, seguida=True))
    F = pd.DataFrame(filas, columns=['foto', 'celula', 'fila_sup', 'fila_inf', 'longitud_px', 'sale_del_canal', 'saltada',
                                     'fila_inf_visible', 'confianza', 'visible', 'seguida'])
    if len(F):
        F = F.sort_values(['foto', 'fila_sup']).reset_index(drop=True)
        F.insert(2, 'orden', F.groupby('foto').cumcount() + 1)
    C = pd.DataFrame([dict(celula=i, madre_id=d['madre'], nace=d['nace'], primera=d['primera'], ultima=d['ultima'],
                           final=d['final'], divide=d['divide']) for i, d in cel.items()],
                     columns=['celula', 'madre_id', 'nace', 'primera', 'ultima', 'final', 'divide'])
    return F, C


# =========================================================================== global tracking (option 3)
"""
Global tracking of the boxes in a channel (option 3): the whole movie at once.

Candidates: every box from the network in every frame and, in addition, the union of each pair of touching
boxes (in case the network splits a cell in two). Possible links between a frame and the following ones (up to
MAX_SALTO frames without seeing the cell): continue (1 -> 1), divide (1 -> 2, with both daughters in the next
frame or one of them a frame later), appear, go away (leave through the exit, disappear inside the channel or
end of the movie) and «false positive» (a box from the network that is not a cell). Each box is
explained exactly once; each cell has a single arrival and a single departure; cells do not
cross. The combination with minimum total cost is chosen with integer programming (HiGHS, scipy.optimize.milp).
The costs are those of the frame-by-frame tracking (expected growth, position pushed by the
cells above, split, network confidence). It is solved twice: the second time with each cell's own
growth rate, estimated from the first solution.

Hand-corrected frames (`manuales`): in the frames of a channel that have your boxes (Cajas_Canales.ijm), your
boxes are used instead of the network's, and they are mandatory: each one is a cell (it cannot be a false positive
or be merged with another one) and its length is always measured (unless it passes the exit). Moreover, inside the
zone you annotated (up to 3 px below your last box; the whole channel if you marked it empty) a cell cannot «skip»
that frame: if you did not give it a box, it was not there.
"""
import time as _time
from scipy.optimize import milp as _milp, LinearConstraint as _LC, Bounds as _Bounds
from scipy import sparse as _sparse

PRM_GLOBAL = dict(
    sig_pos=3.0, sig_pos_k=0.15, sig_libre=5.0, sig_baja=4.0, tope_pos=6.0,
    sig_logL=0.07, sig_Lpx=2.0, tope=12.0,
    contacto=4.0,
    c_div=3.0, c_div3=9.0, sig_reparto=0.12, c_joven=10.0, max_hueco_div=6.0,
    c_falta=3.0,          # for each frame in which a cell is not seen
    c_nueva=16.0,         # appearing in the middle of the movie
    c_ini=6.0,            # appearing near the start: c_ini per frame since the start (the network did not see it)
    c_desap=3.0,          # disappearing inside the channel (death, lysis or loss)
    c_salida=0.5,
    c_fp=12.0,             # box from the network that is not a cell (× confidence)
    c_union=8.0,          # using the union of two touching boxes as a single cell
    horizonte_salida=4,   # frames: a cell that the network stops seeing and that would reach the exit within this time «leaves»
    max_salto=2,          # consecutive frames without seeing a cell that can be skipped
    max_cand=6,           # destinations per candidate and frame (the lowest-cost ones)
    max_coste=25.0,
    z_dudosa=3.0,
    alfa0=0.02, alfa_min=-0.02, alfa_max=0.06,
    max_salto_min=20.0,
    max_vueltas_cruces=6,
    max_dist=80.0,        # px: a continue link cannot move the upper pole down by more than this relative to the expected position (per frame)
    max_dist_sube=25.0,   # px: nor move it up by more than this relative to where it was (plus the rise of the whole channel) if it
                          # passes over other boxes (a lone cell in an almost empty channel can move up further)
    sube_min=4, sube_max=40, sube_mejora=4.0,    # collective upward shift of the channel contents (px, px, improvement in px per box)
    tiempo_max=60.0,      # s per segment (HiGHS)
)


class _Tramo:
    """Candidates for a segment of consecutive frames (without long gaps)."""

    def __init__(self, fotos, cajas, tiempo, fondo, fs, p, manuales=None):
        self.p = p; self.fs = fs; self.fondo = fondo
        self.fotos = fotos                                     # frame indices (0..T-1) of the segment
        manuales = manuales or {}
        self.zona = {k: float(manuales[t]) for k, t in enumerate(fotos) if t in manuales}   # k -> row down to which you annotated
        self.tt = np.array([tiempo[t] for t in fotos], float)
        self.det = []                                          # (k, a, b, conf): k = index within the segment
        for k, t in enumerate(fotos):
            bx = sorted([(max(float(a), fondo - 1.0), float(b), float(c)) for a, b, c in cajas[t] if a < fs - 1 and b > a + 1])
            for a, b, c in bx:
                if b > a + 1: self.det.append((k, a, b, c))
        self.nodos = []                                        # dict(k, a, b, conf, comps)
        por_k = {}
        for j, (k, a, b, c) in enumerate(self.det):
            por_k.setdefault(k, []).append(j)
        self.det_k = por_k
        for k in range(len(fotos)):
            js = por_k.get(k, [])
            for j in js:
                _, a, b, c = self.det[j]; self.nodos.append(dict(k=k, a=a, b=b, conf=c, comps=(j,)))
            for j1, j2 in zip(js[:-1], js[1:]):              # unions of touching boxes (not in your frames)
                if k in self.zona: break
                _, a1, b1, c1 = self.det[j1]; _, a2, b2, c2 = self.det[j2]
                if a2 - b1 <= p['contacto']:
                    self.nodos.append(dict(k=k, a=a1, b=b2, conf=min(c1, c2), comps=(j1, j2), union=True))
        self.nk = {}
        for n, d in enumerate(self.nodos): self.nk.setdefault(d['k'], []).append(n)
        for k in self.nk: self.nk[k].sort(key=lambda n: (self.nodos[n]['a'], self.nodos[n]['b']))
        self.delta = [self._mov_colectivo(k) for k in range(len(fotos) - 1)]

    def _mov_colectivo(self, k):
        """Collective upward shift (px, <= 0) of all the channel contents between frames k and k+1. It is accepted if,
        shifting frame k so that the upper poles of the first box coincide, the relative positions of the first
        boxes fit much better than without moving (same order) or than with a new box at the top."""
        p = self.p; j0, j1 = self.det_k.get(k, []), self.det_k.get(k + 1, [])
        n = min(3, len(j0) - 1, len(j1) - 1)
        if n < 2: return 0.0
        a0 = [self.det[j][1] for j in j0]; a1 = [self.det[j][1] for j in j1]
        if a0[0] - (self.fondo - 1) < p['sube_min']: return 0.0
        jp = self.det_k.get(k - 1, [])                    # the gap at the top was already there before (it is not the network losing the top cell)
        if k > 0 and (not jp or self.det[jp[0]][1] - (self.fondo - 1) < p['sube_min']): return 0.0
        dl = a1[0] - a0[0]
        if dl > -p['sube_min'] or dl < -p['sube_max'] or a1[0] < self.fondo - 4: return 0.0
        cap = 10.0
        cA = sum(min(abs((a1[i] - a1[0]) - (a0[i] - a0[0])), cap) for i in range(n))     # everything moves up by dl
        cB = sum(min(abs(a1[i + 1] - a0[i]), cap) for i in range(n))                     # still, new box at the top
        cC = sum(min(abs(a1[i] - a0[i]), cap) for i in range(n))                         # still
        cD = sum(min(abs(a1[i] - a0[i + 1]), cap) for i in range(n))                     # still, the top one is lost
        if cA + p['sube_mejora'] * n <= min(cB, cC, cD): return float(dl)
        return 0.0

    def manual(self, k):
        return k in self.zona

    def anotada(self, k, a):
        """Is frame k one of yours and is row a inside the zone you annotated?"""
        return k in self.zona and a < self.zona[k]

    def salta_anotada(self, k1, a1, k2, a2):
        """Does a link from (k1, a1) to (k2, a2) skip any of your frames in which the cell would be inside your zone?"""
        for kk in range(k1 + 1, k2):
            if kk in self.zona:
                w = (kk - k1) / (k2 - k1)
                if self.anotada(kk, a1 + w * (a2 - a1)): return True
        return False

    def hay_cajas_entre(self, k, a0, a1):
        """Is there any box in frame k with its upper pole between a0 and a1?"""
        return any(a0 <= self.det[j][1] < a1 for j in self.det_k.get(k, []))

    def subida(self, k, k2, a_nodo):
        """Cumulative collective shift between frames k and k2, without going past the closed end."""
        dl = sum(self.delta[k:k2])
        return max(dl, (self.fondo - 1) - a_nodo) if dl < 0 else 0.0

    def prediccion(self, k, alfa_det):
        """Expected displacement of the upper pole of each box from frame k to frame k+1, not per unit
        of time: returns, for a given dt, a function (n, dt) -> (displacement, free)."""
        js = self.det_k.get(k, [])
        if not js: return {}
        out = {}; acum = 0.0; libre = self.det[js[0]][1] > self.fondo + 6; prev_b = None
        cr = []                                                 # (a, b, alfa) of the boxes
        for j in js:
            _, a, b, _ = self.det[j]; cr.append((a, b, alfa_det.get(j, self.p['alfa0'])))
        return cr, libre

    def llega_salida(self, k, a, b, alfa, alfa_det):
        """Would it reach the exit in the next frames (horizonte_salida), pushed by the cells above?"""
        dtip = float(np.median(np.diff(self.tt))) if len(self.tt) > 1 else 5.0
        dt = self.p['horizonte_salida'] * dtip
        sh, _ = self.desplazamiento(k, a, dt, alfa_det)
        return a + sh + (b - a) * np.exp(alfa * dt) >= self.fs - 1.5

    def desplazamiento(self, k, a_nodo, dt, alfa_det):
        """How much the boxes of frame k that are above a_nodo and touch it (in a chain) grow in dt."""
        js = self.det_k.get(k, []); p = self.p
        cadena = []; libre = True
        for j in js:
            _, a, b, _ = self.det[j]
            if a >= a_nodo - 0.5: break
            if cadena and a - cadena[-1][1] > p['contacto']: cadena = []
            cadena.append((a, b, alfa_det.get(j, p['alfa0'])))
        if cadena and a_nodo - cadena[-1][1] > p['contacto']: cadena = []
        if cadena: libre = cadena[0][0] > self.fondo + 6
        else: libre = a_nodo > self.fondo + 6
        s = sum((b - a) * (np.exp(al * dt) - 1) for a, b, al in cadena)
        return s, libre


def _alfa_ini(mu, t, p):
    if mu is not None and t < len(mu) and np.isfinite(mu[t]): return float(mu[t])
    return p['alfa0']


def _resolver(T, alfa_nodo, alfa_det, L0tip, mu, p, cruces_extra=()):
    """Sets up and solves the integer problem for a segment. Returns the solution (dict of lists)."""
    N = T.nodos; fs = T.fs; nk = T.nk; K = len(T.fotos)
    var = []; cost = []                                        # var: (type, data)

    def add(tipo, datos, c):
        var.append((tipo, datos)); cost.append(c); return len(var) - 1

    z = [add('z', n, (p['c_union'] if N[n].get('union') else 0.0)) for n in range(len(N))]
    fp = [add('fp', j, p['c_fp'] * T.det[j][3]) for j in range(len(T.det))]
    fijas = [fp[j] for j in range(len(T.det)) if T.manual(T.det[j][0])]   # your boxes: always cells
    a_var, d_var = {}, {}
    k_fin = K - 1
    for n, d in enumerate(N):
        k = d['k']
        ca = 0.0 if k == 0 else min(p['c_nueva'], p['c_ini'] * k)
        a_var[n] = add('a', n, ca)
    pred = {}                                                   # n -> (ap, Lp) at the next frame (dt of 1 frame)

    def costes_seguir(n, m, gap):
        dn, dm = N[n], N[m]
        dt = T.tt[dm['k']] - T.tt[dn['k']]
        al = alfa_nodo.get(n, _alfa_ini(mu, T.fotos[dn['k']], p))
        Ln = dn['b'] - dn['a']; Lp = Ln * np.exp(al * dt)
        sh, libre = T.desplazamiento(dn['k'], dn['a'], dt, alfa_det)
        sube = T.subida(dn['k'], dm['k'], dn['a'])
        ap = dn['a'] + sh + sube
        s = p['sig_pos'] + p['sig_pos_k'] * abs(sh) + (p['sig_libre'] if libre else 0.0)
        dd = dm['a'] - ap
        arriba = dn['a'] + sube - dm['a']
        if dd > p['max_dist'] * (1 + gap) or arriba > p['max_dist'] * (1 + gap):   # impossible jumps (the cost is capped)
            return np.inf, ap, Lp, sh, libre
        if arriba > p['max_dist_sube'] * (1 + gap) and T.hay_cajas_entre(dn['k'], dm['a'], dn['a'] - 0.5):
            return np.inf, ap, Lp, sh, libre                   # move up a lot only if it does not pass over other cells
        if dd > 0: s += p['sig_baja']
        c = _cuad(dd / s, p['tope_pos'])
        Lo = dm['b'] - dm['a']; sL = np.hypot(p['sig_logL'], p['sig_Lpx'] / max(Lp, 1.0))
        zz = np.log(max(Lo, 1.0) / Lp) / sL
        if dm['b'] >= fs - 1.5: zz = max(zz, 0.0)
        c += _cuad(zz, p['tope'])
        return c + gap * p['c_falta'], ap, Lp, sh, libre

    # go away
    for n, d in enumerate(N):
        k = d['k']
        if k == k_fin:
            cd = 0.0
        else:
            dt = T.tt[k + 1] - T.tt[k]
            al = alfa_nodo.get(n, _alfa_ini(mu, T.fotos[k], p))
            sh, _ = T.desplazamiento(k, d['a'], dt, alfa_det)
            Lp = (d['b'] - d['a']) * np.exp(al * dt)
            if d['a'] + sh >= fs - 2: cd = 0.0
            elif d['a'] + sh + Lp >= fs - 1.5 or d['b'] >= fs - 1.5: cd = p['c_salida']
            elif T.llega_salida(k, d['a'], d['b'], al, alfa_det): cd = 2 * p['c_salida']   # the network loses it next to the exit
            else: cd = p['c_desap']
        d_var[n] = add('d', n, cd)
    # continue
    seg = []
    for n, d in enumerate(N):
        k = d['k']
        for gap in range(p['max_salto'] + 1):
            k2 = k + 1 + gap
            if k2 > k_fin: break
            cands = []
            for m in nk.get(k2, []):
                if gap and T.salta_anotada(k, d['a'], k2, N[m]['a']): continue    # it was not in your frame
                c, ap, Lp, sh, libre = costes_seguir(n, m, gap)
                if c < p['max_coste']: cands.append((c, m))
            cands.sort()
            for c, m in cands[:p['max_cand'] if gap == 0 else 2]:
                seg.append(add('e', (n, m), c))
    # divide: daughters (m1 above, m2 below) in the next frame; or one of them a frame later
    for n, d in enumerate(N):
        k = d['k']
        if k + 1 > k_fin: continue
        dt = T.tt[k + 1] - T.tt[k]
        al = alfa_nodo.get(n, _alfa_ini(mu, T.fotos[k], p))
        Ln = d['b'] - d['a']; Lp = Ln * np.exp(al * dt)
        sh, libre = T.desplazamiento(k, d['a'], dt, alfa_det)
        ap = d['a'] + sh + T.subida(k, k + 1, d['a'])
        s = p['sig_pos'] + p['sig_pos_k'] * abs(sh) + (p['sig_libre'] if libre else 0.0)
        largo = Lp > 2.5 * L0tip
        joven = Lp < 1.3 * L0tip
        def c_hijas(m1, m2, retraso):
            d1, d2 = N[m1], N[m2]
            if d2['a'] < d1['b'] - 2: return np.inf
            if set(d1['comps']) & set(d2['comps']): return np.inf
            if not retraso and d2['a'] - d1['b'] > p['max_hueco_div']: return np.inf
            L1, L2 = d1['b'] - d1['a'], d2['b'] - d2['a']
            corta = d2['b'] >= fs - 1.5    # the lower one reaches the exit
            r = L1 / (L1 + L2)
            dd = d1['a'] - ap
            ss = s + (p['sig_baja'] if dd > 0 else 0.0)
            c = p['c_div'] + _cuad(dd / ss, p['tope_pos'])
            sr = p['sig_reparto'] * (2.5 if largo else 1.0)
            if not corta: c += min(_cuad((r - 0.5) / sr, 4.0), 4.0)
            Ls = (d2['b'] - d1['a']) if not retraso else (L1 + L2)
            sL = np.hypot(p['sig_logL'], p['sig_Lpx'] / max(Lp, 1.0))
            zz = np.log(max(Ls, 1.0) / Lp) / sL
            if corta: zz = max(zz, 0.0)
            c += _cuad(zz, p['tope'])
            if joven: c += p['c_joven']
            if retraso: c += p['c_falta']
            return c
        hijos = nk.get(k + 1, [])
        cand = []
        for i1, m1 in enumerate(hijos):
            if abs(N[m1]['a'] - ap) > 40: continue
            for m2 in hijos[i1 + 1:]:
                if N[m2]['a'] - N[m1]['b'] > p['max_hueco_div']: break
                c = c_hijas(m1, m2, False)
                if c < p['max_coste']: cand.append((c, m1, m2))
        cand.sort()
        for c, m1, m2 in cand[:3]:
            seg.append(add('h', (n, m1, m2), c))
        # one daughter is seen in k+1 and the other in k+2
        if k + 2 <= k_fin:
            cand = []
            for m1 in hijos:
                for m2 in nk.get(k + 2, []):
                    for (u, w) in ((m1, m2), (m2, m1)):
                        du, dw = N[u], N[w]
                        if du['a'] > dw['a']: continue
                        if abs(du['a'] - ap) > 40: continue
                        if T.anotada(k + 1, N[m2]['a']): continue     # the daughter missing in k+1 would have to be in your frame
                        c = c_hijas(u, w, True)
                        if c < p['max_coste']: cand.append((c, u, w))
            cand.sort()
            for c, u, w in cand[:2]:
                seg.append(add('h', (n, u, w), c))
    nv = len(var)
    # ---- constraints
    filas, cols, vals, lo, hi = [], [], [], [], []
    r = [0]

    def fila(coefs, l, h):
        for ci, v in coefs: filas.append(r[0]); cols.append(ci); vals.append(v)
        lo.append(l); hi.append(h); r[0] += 1
    # each box is explained once
    cubre = {j: [] for j in range(len(T.det))}
    for n, d in enumerate(N):
        for j in d['comps']: cubre[j].append(z[n])
    for j in range(len(T.det)):
        fila([(v, 1.0) for v in cubre[j]] + [(fp[j], 1.0)], 1, 1)
    ent = {n: [a_var[n]] for n in range(len(N))}; sal = {n: [d_var[n]] for n in range(len(N))}
    for vi in seg:
        tipo, dat = var[vi]
        if tipo == 'e':
            n, m = dat; sal[n].append(vi); ent[m].append(vi)
        else:
            n, m1, m2 = dat; sal[n].append(vi); ent[m1].append(vi); ent[m2].append(vi)
    for n in range(len(N)):
        fila([(z[n], 1.0)] + [(v, -1.0) for v in ent[n]], 0, 0)
        fila([(z[n], 1.0)] + [(v, -1.0) for v in sal[n]], 0, 0)
    if cruces_extra:                                           # pairs of links (by key) that cannot be chosen together
        idx = {v: i for i, v in enumerate(var)}
        for (u, w) in cruces_extra:
            if u in idx and w in idx: fila([(idx[u], 1.0), (idx[w], 1.0)], -np.inf, 1)
    A = _sparse.csr_matrix((vals, (filas, cols)), shape=(r[0], nv))
    ub = np.ones(nv); ub[fijas] = 0
    res = _milp(c=np.array(cost), constraints=_LC(A, lo, hi), integrality=np.ones(nv),
               bounds=_Bounds(0, ub), options=dict(time_limit=p['tiempo_max'], disp=False))
    if res.x is None:
        raise RuntimeError(f'global tracking: no solution ({res.message})')
    x = np.round(res.x).astype(int)
    return var, x, res


def _cruces(T, var, x):
    """Pairs of chosen links that cross (between the same two frames), as keys (type, data): the variable
    indices change between the 1st and the 2nd pass."""
    N = T.nodos; elegidos = []
    for vi, (tipo, dat) in enumerate(var):
        if x[vi] and tipo in ('e', 'h'):
            if tipo == 'e':
                n, m = dat; elegidos.append((vi, N[n]['k'], N[m]['k'], N[n]['a'], N[m]['a'], N[m]['b']))
            else:
                n, m1, m2 = dat
                elegidos.append((vi, N[n]['k'], N[m1]['k'], N[n]['a'], N[m1]['a'], N[m1]['b']))
                if N[m2]['k'] == N[m1]['k']:
                    elegidos.append((vi, N[n]['k'], N[m2]['k'], N[n]['a'], N[m2]['a'], N[m2]['b']))
    out = []
    for i in range(len(elegidos)):
        for j in range(i + 1, len(elegidos)):
            u, w = elegidos[i], elegidos[j]
            if u[0] == w[0] or u[1] != w[1] or u[2] != w[2]: continue
            if (u[3] - w[3]) * (u[4] - w[4]) < 0: out.append((var[u[0]], var[w[0]]))
    return out


def seguir_cajas_global(cajas, tiempo, fila_fondo, fila_salida, mu=None, malas=(), prm=None, info=None, manuales=None):
    """Like linaje.seguir_cajas, but solving the whole movie at once. Returns (F, C).
    manuales: {frame (from 0): row down to which you annotated} for the frames with your boxes (already put in `cajas`)."""
    p = dict(PRM_GLOBAL, **(prm or {}))
    T = len(cajas); tiempo = np.asarray(tiempo, float); fs = float(fila_salida)
    mala = np.zeros(T, bool); mala[[f - 1 for f in malas if 1 <= f <= T]] = True
    ok = [t for t in range(T) if not mala[t]]
    dt_tip = float(np.median(np.diff(tiempo[ok]))) if len(ok) > 2 else 5.0
    tramos = []; cur = []
    for t in ok:
        if cur and tiempo[t] - tiempo[cur[-1]] > max(p['max_salto_min'], 3 * dt_tip):
            tramos.append(cur); cur = []
        cur.append(t)
    if cur: tramos.append(cur)
    Ls = [b - a for t in ok for a, b, _ in cajas[t] if a < fs - 1 and b < fs - 1.5]
    L0tip = float(np.percentile(Ls, 25)) if len(Ls) > 10 else 20.0
    filas, celdas = [], []; sig = [1]; t0 = _time.time(); nvar = 0; dudas = []
    for it_tr, fotos in enumerate(tramos):
        ultimo_tramo = it_tr == len(tramos) - 1
        Tr = _Tramo(fotos, cajas, tiempo, float(fila_fondo), fs, p, manuales)
        if not Tr.nodos: continue
        alfa_nodo, alfa_det = {}, {}
        cr = []
        for it in range(2):
            for _ in range(p["max_vueltas_cruces"]):          # crossings: they are added and the problem is solved again
                var, x, res = _resolver(Tr, alfa_nodo, alfa_det, L0tip, mu, p, cr)
                nuevos = _cruces(Tr, var, x)
                if not nuevos: break
                cr = cr + nuevos
            nvar = max(nvar, len(var))
            pistas = _pistas(Tr, var, x)
            if it == 0:                                        # each cell's own rate, for the 2nd pass
                for pi in pistas:
                    ks = [Tr.nodos[n]['k'] for n in pi['nodos']]
                    Ls_ = [Tr.nodos[n]['b'] - Tr.nodos[n]['a'] for n in pi['nodos']]
                    ok_ = [i for i, n in enumerate(pi['nodos']) if Tr.nodos[n]['b'] < fs - 1.5]
                    if len(ok_) >= 3:
                        al = np.polyfit(Tr.tt[[ks[i] for i in ok_]], np.log([Ls_[i] for i in ok_]), 1)[0]
                        al = float(np.clip(al, p['alfa_min'], p['alfa_max']))
                        for n in pi['nodos']:
                            alfa_nodo[n] = al
                            for j in Tr.nodos[n]['comps']: alfa_det[j] = al
        _salida(Tr, var, x, pistas, filas, celdas, sig, fs, p, mu, ultimo_tramo)
        for vi, (tipo, dat) in enumerate(var):                 # detector doubts according to the tracking
            if not x[vi]: continue
            if tipo == 'fp' and Tr.det[dat][3] >= 0.5:
                dudas.append((Tr.fotos[Tr.det[dat][0]] + 1, 2, 'la red ve una caja que el seguimiento descarta'))
            elif tipo == 'z' and Tr.nodos[dat].get('union'):
                dudas.append((Tr.fotos[Tr.nodos[dat]['k']] + 1, 3, 'la red parte una célula en dos'))
    if info is not None:
        info.update(segundos=_time.time() - t0, variables=nvar, dudas=dudas)
    F = pd.DataFrame(filas, columns=['foto', 'celula', 'fila_sup', 'fila_inf', 'longitud_px', 'sale_del_canal', 'saltada',
                                     'fila_inf_visible', 'confianza', 'visible', 'seguida', 'interpolada', 'manual'])
    if len(F):
        F = F.sort_values(['foto', 'fila_sup']).reset_index(drop=True)
        F.insert(2, 'orden', F.groupby('foto').cumcount() + 1)
    C = pd.DataFrame(celdas, columns=['celula', 'madre_id', 'nace', 'primera', 'ultima', 'final', 'divide'])
    if info is not None and len(F):
        for r in F[F.interpolada].itertuples():
            dudas.append((int(r.foto), 3, 'la red no ve una célula que estaba'))
        for r in F[~F.interpolada & F.longitud_px.isna() & ~F.sale_del_canal].itertuples():
            dudas.append((int(r.foto), 1, 'caja que no cuadra con la longitud esperada'))
        for r in C.itertuples():
            if r.final == 'division': dudas.append((int(r.divide) + 1, 1, 'división'))
            if r.final == 'desaparece': dudas.append((int(r.ultima) + 1, 2, 'desaparece dentro del canal'))
        info['dudas'] = dudas
    return F, C


def _pistas(T, var, x):
    """Trajectories (one per cell) from the solution: list of dict(nodos, madre_pista, final, nace_k)."""
    N = T.nodos
    sig_e, hijas, inicio = {}, {}, []
    activos = set()
    for vi, (tipo, dat) in enumerate(var):
        if not x[vi]: continue
        if tipo == 'z': activos.add(dat)
        elif tipo == 'e': sig_e[dat[0]] = dat[1]
        elif tipo == 'h': hijas[dat[0]] = (dat[1], dat[2])
        elif tipo == 'a': inicio.append(dat)
    pistas = []; pista_de = {}

    def seguir(n0, madre, nace_k):
        nodos = [n0]; n = n0
        while n in sig_e:
            n = sig_e[n]; nodos.append(n)
        pi = dict(nodos=nodos, madre=madre, nace_k=nace_k, divide=n in hijas)
        pistas.append(pi); idx = len(pistas) - 1
        for m in nodos: pista_de[m] = idx
        if n in hijas:
            for h in hijas[n]: seguir(h, idx, N[n]['k'] + 1)
    for n in sorted(inicio, key=lambda n: (N[n]['k'], N[n]['a'])):
        seguir(n, None, None)
    return pistas


def _salida(T, var, x, pistas, filas, celdas, sig, fs, p, mu, ultimo_tramo=True):
    N = T.nodos; base = sig[0]; K = len(T.fotos)
    ids = {i: base + i for i in range(len(pistas))}; sig[0] += len(pistas)
    ocup = {}
    for pi in pistas:
        for n in pi['nodos']: ocup.setdefault(N[n]['k'], []).append((N[n]['a'], N[n]['b']))
    for i, pi in enumerate(pistas):
        nod = pi['nodos']; last = N[nod[-1]]
        # cell that appears shortly after the start: the network did not see it in the first frames; they are filled in backwards
        d0 = N[nod[0]]; k_primera = d0['k']
        # the network did not see it in its first frames (at the start of the segment, or a daughter seen one frame
        # later than her sister): boxes filled in backwards, without length
        k_desde = None
        if pi['madre'] is None and 1 <= d0['k'] <= p['max_salto']: k_desde = 0
        elif pi['nace_k'] is not None and pi['nace_k'] < d0['k']: k_desde = pi['nace_k']
        if k_desde is not None:
            for kk in range(k_desde, d0['k']):
                L = (d0['b'] - d0['a']) * np.exp(-p['alfa0'] * (T.tt[d0['k']] - T.tt[kk])); a, b = d0['a'], d0['a'] + L
                if any(min(b, y) - max(a, x) > 0.3 * L for x, y in ocup.get(kk, [])): continue
                if T.anotada(kk, a): continue                     # it was not in your frame
                k_primera = min(k_primera, kk)
                filas.append(dict(foto=T.fotos[kk] + 1, celula=ids[i], fila_sup=round(a, 2), fila_inf=round(b, 2),
                                  longitud_px=np.nan, sale_del_canal=False, saltada=False, fila_inf_visible=round(min(b, fs - 1), 2),
                                  confianza=np.nan, visible=True, seguida=True, interpolada=True, manual=False))
        if pi['divide']: final = 'division'
        elif last['k'] >= K - 1 - p['max_salto']:                     # at the end of the segment it cannot be known
            final = 'fin' if ultimo_tramo else 'perdida'                  # (perdida: a long gap of bad frames follows)
        elif last['b'] >= fs - 1.5: final = 'sale'
        else:
            final = 'sale' if T.llega_salida(last['k'], last['a'], last['b'], p['alfa0'], {}) else 'desaparece'
        hijo_k = [pistas[j]['nace_k'] for j in range(len(pistas)) if pistas[j]['madre'] == i]
        celdas.append(dict(celula=ids[i], madre_id=ids[pi['madre']] if pi['madre'] is not None else None,
                           nace=T.fotos[pi['nace_k']] if pi['nace_k'] is not None else np.nan,
                           primera=T.fotos[k_primera], ultima=T.fotos[last['k']], final=final,
                           divide=T.fotos[min(hijo_k)] if hijo_k else np.nan))
        # lengths: doubtful if far from the cell's own trend
        ks = np.array([N[n]['k'] for n in nod]); L = np.array([N[n]['b'] - N[n]['a'] for n in nod])
        sale = np.array([N[n]['b'] >= fs - 1.5 for n in nod])
        fiable = np.ones(len(nod), bool)
        if (~sale).sum() >= 4:
            tt = T.tt[ks]; m = ~sale
            res = np.log(L[m]) - np.polyval(np.polyfit(tt[m], np.log(L[m]), 1), tt[m])
            sL = np.hypot(p['sig_logL'], p['sig_Lpx'] / np.maximum(L[m], 1.0))
            fiable[np.where(m)[0][np.abs(res) > p['z_dudosa'] * sL]] = False
        man = np.array([T.manual(N[n]['k']) for n in nod]); fiable[man] = True     # your boxes: always measured
        for q, n in enumerate(nod):
            d = N[n]; s = bool(sale[q])
            filas.append(dict(foto=T.fotos[d['k']] + 1, celula=ids[i], fila_sup=round(d['a'], 2), fila_inf=round(d['b'], 2),
                              longitud_px=np.nan if (s or not fiable[q]) else round(d['b'] - d['a'], 2), sale_del_canal=s,
                              saltada=False, fila_inf_visible=round(min(d['b'], fs - 1), 2), confianza=round(d['conf'], 3),
                              visible=True, seguida=True, interpolada=False, manual=bool(man[q])))
            if q + 1 < len(nod):                               # frames where it is not seen: interpolated box, without length
                d2 = N[nod[q + 1]]
                for kk in range(d['k'] + 1, d2['k']):
                    w = (kk - d['k']) / (d2['k'] - d['k'])
                    a = d['a'] + w * (d2['a'] - d['a']); b = d['b'] + w * (d2['b'] - d['b'])
                    filas.append(dict(foto=T.fotos[kk] + 1, celula=ids[i], fila_sup=round(a, 2), fila_inf=round(b, 2),
                                      longitud_px=np.nan, sale_del_canal=bool(b >= fs - 1.5), saltada=False,
                                      fila_inf_visible=round(min(b, fs - 1), 2), confianza=np.nan, visible=True,
                                      seguida=True, interpolada=True, manual=False))


def tablas(F, C, canal, tiempo):
    """
    Output tables for a channel: per cell and frame (length, position…) and per cell (cycle:
    birth, division or exit, L0, Ld, rate, split and type of division it was born from).
    """
    tiempo = np.asarray(tiempo, float)
    F = F.copy(); F.insert(0, 'canal', canal); F.insert(2, 'tiempo_min', tiempo[F.foto.values - 1])
    filas = []
    por_c = {c: g.sort_values('foto') for c, g in F.groupby('celula')}
    hijos = C.groupby('madre_id').celula.apply(list).to_dict() if C.madre_id.notna().any() else {}
    gen = {}
    def generacion(c):
        if c in gen: return gen[c]
        m = C.loc[C.celula == c, 'madre_id'].iloc[0]
        gen[c] = 0 if pd.isna(m) else generacion(int(m)) + 1
        return gen[c]
    for r in C.itertuples():
        g = por_c.get(r.celula)
        if g is None or not len(g): continue
        nace = not pd.isna(r.nace); divide = r.final == 'division'
        v = g[g.longitud_px.notna() & ~g.saltada]
        tasa = np.polyfit(v.tiempo_min, np.log(v.longitud_px), 1)[0] if len(v) >= 3 and (v.longitud_px > 0).all() else np.nan
        # at birth and at division: the first / last measured length (if it could not be measured in the first frame,
        # e.g. a filled-in box, the next one is used, up to 2 frames)
        L0 = next((v for v in g.longitud_px.iloc[:3] if v == v), np.nan) if nace else np.nan
        Ld = next((v for v in g.longitud_px.iloc[::-1].iloc[:3] if v == v), np.nan) if divide else np.nan
        reparto, tipo = np.nan, ''
        if nace and not pd.isna(r.madre_id):
            hs = hijos.get(r.madre_id, [])
            if len(hs) >= 3: tipo = 'fragmentación'
            else:
                hermana = [h for h in hs if h != r.celula]
                if hermana and hermana[0] in por_c:
                    Lh = next((v for v in por_c[hermana[0]].longitud_px.iloc[:3] if v == v), np.nan)
                    if not np.isnan(L0) and not np.isnan(Lh) and L0 + Lh > 0:
                        reparto = L0 / (L0 + Lh); tipo = 'simétrica' if 0.3 <= reparto <= 0.7 else 'polar'
        filas.append(dict(canal=canal, celula=r.celula, madre=r.madre_id, generacion=generacion(r.celula),
                          orden_al_nacer=int(g.orden.iloc[0]),
                          nace_foto=int(r.nace) + 1 if nace else np.nan, nace_min=tiempo[int(r.nace)] if nace else np.nan,
                          divide_foto=int(r.divide) + 1 if divide else np.nan, divide_min=tiempo[int(r.divide)] if divide else np.nan,
                          final={'division': 'se divide', 'sale': 'sale o deja de verse', 'fin': 'fin de la película',
                                 'desaparece': 'desaparece en el canal (muere, se lisa o se pierde)',
                                 'perdida': 'se pierde (hueco largo de fotos malas)'}[r.final],
                          duracion_min=(tiempo[int(r.divide)] - tiempo[int(r.nace)]) if nace and divide else np.nan,
                          L0_px=L0, Ld_px=Ld, tasa_por_min=tasa, reparto=reparto, nace_por=tipo,
                          completo=bool(nace and divide), fotos=len(g)))
    return F, pd.DataFrame(filas)


COLORES = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
           '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']


def figura(ruta_img, canal, fotos=None, salida=None):
    """
    Review figure of the lineage of a channel: the frames in a row, one box per cell with its number
    (same colour while it is the same cell) and, at birth, S / P / F (symmetric, polar or
    fragmentation division). It is saved as <image>_linaje_cNN.png.
    """
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from pathlib import Path
    import detectar_celulas as DC
    ruta_img = Path(ruta_img); pref = ruta_img.with_suffix('')
    F = pd.read_csv(f'{pref}_linaje.csv'); C = pd.read_csv(f'{pref}_linaje_ciclos.csv')
    F = F[F.canal == canal]; C = C[C.canal == canal].set_index('celula')
    img, geo = DC.abrir(ruta_img); k = int(np.where(geo.canal.values == canal)[0][0]); g = geo.iloc[k]
    if fotos is None: fotos = sorted(F.foto.unique())
    fotos = [f for f in fotos if f in set(F.foto)]
    til = DC.tiles(img, k, int(g.ancho), int(g.sep), int(max(fotos))); W = til.shape[2]
    por_fila = 30
    filas = [fotos[i:i + por_fila] for i in range(0, len(fotos), por_fila)]
    fig, axs = plt.subplots(len(filas), 1, figsize=(min(len(fotos), por_fila) * 0.62 + 1, 4.2 * len(filas)), squeeze=False)
    for ax, fr in zip(axs[:, 0], filas):
        mos = np.concatenate([np.pad(til[f - 1], ((0, 0), (0, 4)), constant_values=til.max()) for f in fr], 1)
        ax.imshow(mos, cmap='gray', aspect='auto')
        for n, f in enumerate(fr):
            x0 = n * (W + 4)
            for r in F[F.foto == f].itertuples():
                col = COLORES[r.celula % len(COLORES)]
                fin = min(r.fila_inf, g.fila_salida + 3)
                ax.add_patch(plt.Rectangle((x0 + 2, r.fila_sup + 0.5), W - 4, fin - r.fila_sup, fill=False, ec=col, lw=1.3,
                                           ls='--' if r.sale_del_canal else '-'))   # dashed: reaches the exit (or the end is not visible)
                ax.text(x0 + W - 2, r.fila_sup + 7, str(r.celula), color=col, fontsize=6, ha='right', weight='bold')
                if r.celula in C.index and C.loc[r.celula, 'nace_foto'] == f:
                    ax.text(x0 + 3, r.fila_sup + 7, {'polar': 'P', 'simétrica': 'S', 'fragmentación': 'F'}.get(C.loc[r.celula, 'nace_por'], '·'),
                            color='white', fontsize=7, weight='bold')
            ax.text(x0 + 2, -4, str(f), fontsize=7)
        ax.set_ylim(g.fila_salida + 8, -8); ax.axis('off')
    axs[0, 0].set_title(f'{ruta_img.stem} · channel {canal}: one box per cell (number = cell); at birth S/P/F = symmetric/polar/fragmentation division; dashed = end not visible', fontsize=9)
    plt.tight_layout()
    salida = salida or f'{pref}_linaje_c{canal:02d}.png'
    plt.savefig(salida, dpi=80); plt.close(fig)
    return salida
