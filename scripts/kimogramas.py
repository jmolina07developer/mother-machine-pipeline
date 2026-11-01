"""
Kymographs from a mother machine: detects the channels in an aligned stack and,
for each channel, stacks over time the intensity profile along its axis.

A kymograph has the position along the channel on the vertical axis (row 0 =
MARGEN_SUP px above the closed end; the channel exit is marked) and time on the
horizontal axis (one column per slice).

Saves to <stack folder>/Kimogramas/:
    <base>_kimogramas.npz     everything together for Python/AI: raw bright-field
                              kymographs (channel, position, time), brightness of each slice,
                              channel geometry, timestamps
    <base>_kimo_BF.tif        one slice per channel, to view and trace in Fiji
                              (Trazar_Kimogramas.ijm): 8-bit, without the flicker between slices,
                              with the closed end and the exit (yellow) and your measurements (red)
    <base>_canales.csv        geometry of each channel (axis, closed end, exit) and,
                              if there is a RoiSet, which user channel it is
    <base>_kimo_etiquetas.csv your measurements mapped onto the kymograph (row of the upper
                              and lower pole of each line, per slice)
    <base>_kimogramas.png     all channels with your measurements on top, for review

Channels are numbered from LEFT to RIGHT (in the macro you used to number them from
right to left; the correspondence is in _canales.csv).

Usage:
    python kimogramas.py 01-02102026-BF.tif            (uses the RoiSet if it exists)
    python kimogramas.py Stacks/*-BF.tif               (several at once)

Requires: pip install tifffile scikit-image scipy pandas matplotlib roifile
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks

MARGEN_SUP = 10    # px above the closed end included in the kymograph
EXTRA_INF = 40     # px below the channel exit (to see the cells leaving)
ANCHO = 5          # px averaged perpendicular to the axis (inside the channel)


# ---------------------------------------------------------------- geometry
def _autocorr_periodo(perfil, lag_min=20, lag_max=80):
    p = perfil - perfil.mean()
    ac = np.correlate(p, p, mode='full')[len(p) - 1:]
    ac = ac / ac[0]
    lags = np.arange(lag_min, min(lag_max, len(ac) - 1))
    k = lags[np.argmax(ac[lags])]
    return int(k), float(ac[k])


def detectar_canales(media, periodo_min=20, periodo_max=80):
    """
    media : mean (or median) image of the aligned stack.
    Returns (canales, info): DataFrame with one row per channel (left -> right):
    x_sup, y_sup (closed end), x_inf, y_inf (exit), and a dict with period and band.
    """
    M = gaussian_filter(media.astype(float), 1)
    gx = np.abs(np.diff(M, axis=1))                      # vertical edges (walls)
    energia = gaussian_filter1d(gx.mean(1), 3)

    # rows with periodic structure (channels) versus numbers / background
    periodicidad = np.zeros(M.shape[0])
    for y in range(0, M.shape[0], 4):
        fila = M[max(0, y - 3):y + 4].mean(0)
        periodicidad[y:y + 4] = _autocorr_periodo(fila, periodo_min, periodo_max)[1]
    es_canal = (periodicidad > 0.4) & (energia > 0.3 * energia.max())
    # longest contiguous band
    mejor, ini = (0, 0, 0), None
    for y, v in enumerate(np.r_[es_canal, False]):
        if v and ini is None:
            ini = y
        if not v and ini is not None:
            if y - ini > mejor[0]:
                mejor = (y - ini, ini, y)
            ini = None
    _, y0, y1 = mejor
    if y1 - y0 < 50:
        raise RuntimeError('Cannot find the band of channels')

    centro = slice(y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4)
    perfil = M[centro].mean(0)
    periodo, _ = _autocorr_periodo(perfil, periodo_min, periodo_max)
    suave = gaussian_filter1d(perfil, periodo / 7)
    minimos, props = find_peaks(-suave, distance=int(0.6 * periodo), prominence=0)
    prom = props['prominences']
    ref = np.median(np.sort(prom)[-max(3, len(prom) // 2):])
    minimos = minimos[prom > 0.3 * ref]
    margen = periodo // 2
    # channels cut by the image border: extend the comb if the axis fits and
    # there are walls in that region (profile with contrast similar to the other channels)
    def contraste(x):
        return float(np.std(perfil[max(0, x - margen):min(len(perfil), x + margen)]))
    contraste_tipico = np.median([contraste(x) for x in minimos])
    extra = []
    for x_pred in (minimos.min() - periodo, minimos.max() + periodo):
        lo, hi = max(int(x_pred - periodo // 4), 0), min(int(x_pred + periodo // 4) + 1, M.shape[1])
        if hi - lo < 3:
            continue
        x = lo + int(np.argmin(suave[lo:hi]))
        if x in (lo, hi - 1):                    # minimum stuck to the border: trust the period
            x = int(round(x_pred))
        if (2 <= x <= M.shape[1] - 2 and suave[x] < np.median(suave)
                and contraste(x) > 0.5 * contraste_tipico):
            extra.append(x)
    minimos = np.sort(np.r_[minimos, extra]).astype(int)
    minimos = minimos[(minimos >= 2) & (minimos <= M.shape[1] - 2)]

    filas = []
    ys = np.arange(y0 + 5, y1 - 5, 8)
    for xc in minimos:
        # axis: channel centre at several heights -> line x = a + b·y
        xs = []
        for y in ys:
            f = gaussian_filter1d(M[y - 3:y + 4].mean(0), periodo / 7)
            lo, hi = max(int(xc - periodo // 3), 0), min(int(xc + periodo // 3) + 1, M.shape[1])
            xs.append(lo + np.argmin(f[lo:hi]))
        b, a = np.polyfit(ys, xs, 1)
        if abs(b) > 0.1:                                 # absurd tilt: vertical
            a, b = float(np.median(xs)), 0.0
        # ends: where there are walls around the axis
        lo, hi = max(int(xc - margen), 0), min(int(xc + margen), gx.shape[1])
        e = gaussian_filter1d(gx[:, lo:hi].mean(1), 2)
        nivel = 0.5 * np.median(e[y0:y1])
        dentro = np.where(e > nivel)[0]
        dentro = dentro[(dentro > y0 - 40) & (dentro < y1 + 40)]
        ys_c, ye_c = (dentro.min(), dentro.max()) if len(dentro) else (y0, y1)
        filas.append(dict(x_sup=a + b * ys_c, y_sup=float(ys_c), x_inf=a + b * ye_c, y_inf=float(ye_c),
                          al_borde=bool(xc < margen or xc > M.shape[1] - margen)))
    canales = pd.DataFrame(filas)
    # The closed ends and the exits of all channels lie on two straight lines (the chip is
    # rigid): fit them and correct the channels that deviate (e.g. a dark cell
    # that fools the exit detector).
    for xcol, ycol in (('x_sup', 'y_sup'), ('x_inf', 'y_inf')):
        x, y = canales[xcol].to_numpy(), canales[ycol].to_numpy()
        ok = np.ones(len(x), bool)
        for _ in range(3):
            pend, ord_ = np.polyfit(x[ok], y[ok], 1) if ok.sum() >= 3 else (0.0, np.median(y))
            res = y - (pend * x + ord_)
            ok = np.abs(res) <= max(3.0, 2.5 * np.median(np.abs(res[ok])))
        malos = np.abs(res) > 6
        canales.loc[malos, ycol] = pend * x[malos] + ord_
        canales['corregido_' + ycol[2:]] = malos
    # recompute x at the corrected ends following each channel's axis
    for i, f in enumerate(filas):
        b = (f['x_inf'] - f['x_sup']) / (f['y_inf'] - f['y_sup']) if f['y_inf'] != f['y_sup'] else 0.0
        canales.loc[i, 'x_sup'] = f['x_sup'] + b * (canales.loc[i, 'y_sup'] - f['y_sup'])
        canales.loc[i, 'x_inf'] = f['x_sup'] + b * (canales.loc[i, 'y_inf'] - f['y_sup'])
    canales.insert(0, 'canal', np.arange(1, len(canales) + 1))
    canales['longitud_px'] = np.hypot(canales.x_inf - canales.x_sup, canales.y_inf - canales.y_sup)
    return canales, dict(periodo=periodo, banda=(int(y0), int(y1)))


# ---------------------------------------------------------------- kymographs
def kimograma(stack, canal, ancho=5, margen_sup=10, extra_inf=40, largo=None, reducir='media'):
    """
    Samples `stack` (T, Y, X) along the channel axis, with `ancho` px perpendicular to it.
    reducir: 'media' (transverse mean), 'minimo' or 'desv' (transverse standard deviation;
    a septum darkens the whole width of the cell, a spot only part of it).
    Returns K (L, T): row 0 = margen_sup px above the closed end.
    """
    p0 = np.array([canal.x_sup, canal.y_sup])
    p1 = np.array([canal.x_inf, canal.y_inf])
    u = (p1 - p0) / np.linalg.norm(p1 - p0)              # along the channel
    n = np.array([-u[1], u[0]])                          # perpendicular
    if largo is None:
        largo = int(round(np.linalg.norm(p1 - p0))) + margen_sup + extra_inf
    s = np.arange(largo) - margen_sup
    w = np.arange(ancho) - (ancho - 1) / 2
    pts = p0[None, None, :] + s[:, None, None] * u + w[None, :, None] * n   # (L, W, 2) in x,y
    filas_img, cols_img = pts[..., 1], pts[..., 0]
    red = {'media': np.mean, 'minimo': np.min, 'desv': np.std}[reducir]
    K = np.empty((largo, len(stack)), dtype=np.float32)
    for t, img in enumerate(stack):
        v = map_coordinates(img.astype(np.float32), [filas_img.ravel(), cols_img.ravel()],
                            order=1, mode='nearest').reshape(largo, ancho)
        K[:, t] = red(v, axis=1)
    return K


def proyectar_lineas(medidas, canales, periodo, margen_sup=10):
    """
    Maps the measured lines to kymograph coordinates: for each measurement, the nearest
    detected channel and the position (row) of its two ends along the axis.
    """
    filas = []
    for _, m in medidas.iterrows():
        pts = np.asarray(m['coordenadas'], float) - 0.5      # ImageJ -> pixel centre
        mid = pts.mean(0)
        mejor = None
        for _, c in canales.iterrows():
            p0 = np.array([c.x_sup, c.y_sup]); p1 = np.array([c.x_inf, c.y_inf])
            u = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-u[1], u[0]])
            d = abs((mid - p0) @ n)
            if mejor is None or d < mejor[0]:
                mejor = (d, c.canal, p0, u)
        d, canal, p0, u = mejor
        if d > periodo / 2:
            continue
        s = (pts - p0) @ u
        filas.append(dict(canal=int(canal), canal_usuario=int(m['canal']), slice=int(m['slice']),
                          fila_sup=float(s.min() + margen_sup), fila_inf=float(s.max() + margen_sup),
                          distancia_eje=float(d)))
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- all together
def _horas(etiquetas):
    patron = re.compile(r'_(\d{6})_(\d{6})_')
    out = []
    for e in etiquetas:
        m = patron.search(e or '')
        out.append(pd.to_datetime(m.group(1) + m.group(2), format='%y%m%d%H%M%S') if m else pd.NaT)
    return pd.Series(out)


def normalizar(K):
    """Removes the flicker between slices (divides each column by its median) and scales to 8 bits."""
    k = K / np.median(K, axis=0, keepdims=True)
    lo, hi = np.percentile(k, (0.5, 99.8))
    return np.clip((k - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _superposicion(canales, etiq, largo, T):
    """ImageJ ROIs (bytes) shown over the kymograph in Fiji: closed end, exit and your measurements."""
    import roifile
    rois = []
    for i, c in canales.iterrows():
        pos = i + 1
        for fila, color in ((c.fila_fondo, b'\xff\xed\xa1\x00'), (c.fila_salida, b'\xff\xed\xa1\x00')):
            r = roifile.ImagejRoi.frompoints(np.array([[0, fila + 0.5], [T, fila + 0.5]], dtype=np.float32))
            r.roitype = roifile.ROI_TYPE.LINE
            r.x1, r.y1, r.x2, r.y2 = 0.0, fila + 0.5, float(T), fila + 0.5
            r.position, r.stroke_color, r.name = pos, color, ('fondo' if fila == c.fila_fondo else 'salida')
            rois.append(r.tobytes())
        if len(etiq):
            e = etiq[etiq.canal == c.canal]
            if len(e):
                pts = np.r_[np.c_[e.slice - 0.5, e.fila_inf + 0.5], np.c_[e.slice - 0.5, e.fila_sup + 0.5]]
                r = roifile.ImagejRoi.frompoints(pts.astype(np.float32))
                r.roitype = roifile.ROI_TYPE.POINT
                r.position, r.stroke_color, r.name = pos, b'\xff\xe3\x49\x48', 'medidas'
                rois.append(r.tobytes())
    return rois


def generar_kimogramas(ruta_bf, ruta_roiset=None, salida=None, buscar=True, figura=True):
    """
    ruta_bf     : ALIGNED bright-field stack (…-BF.tif or …_alineado.tif)
    ruta_roiset : your measurements (by default <stack>_RoiSet.zip if it exists)
    """
    ruta_bf = Path(ruta_bf)
    base = ruta_bf.stem.replace('-BF', '')
    if buscar and ruta_roiset is None and ruta_bf.with_name(ruta_bf.stem + '_RoiSet.zip').exists():
        ruta_roiset = ruta_bf.with_name(ruta_bf.stem + '_RoiSet.zip')
    salida = Path(salida) if salida else ruta_bf.parent / 'Kimogramas'
    salida.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(ruta_bf) as tif:
        bf = tif.asarray()
        etiquetas = (tif.imagej_metadata or {}).get('Labels') or [''] * len(bf)
    etiquetas = [e.split('\n')[0].strip() for e in etiquetas]
    canales, info = detectar_canales(np.median(bf, axis=0))
    largo = int(np.ceil(canales.longitud_px.max())) + MARGEN_SUP + EXTRA_INF
    canales['fila_fondo'] = MARGEN_SUP
    canales['fila_salida'] = MARGEN_SUP + canales['longitud_px']

    K = np.stack([kimograma(bf, c, ANCHO, MARGEN_SUP, EXTRA_INF, largo) for _, c in canales.iterrows()])
    # auxiliary kymographs for segmentation: minimum and deviation across a wider
    # strip (almost the whole channel)
    ancho_aux = max(ANCHO, int(0.25 * info['periodo']) | 1)
    K_min = np.stack([kimograma(bf, c, ancho_aux, MARGEN_SUP, EXTRA_INF, largo, 'minimo') for _, c in canales.iterrows()])
    K_sd = np.stack([kimograma(bf, c, ancho_aux, MARGEN_SUP, EXTRA_INF, largo, 'desv') for _, c in canales.iterrows()])
    brillo = np.median(bf.reshape(len(bf), -1)[:, ::7], axis=1).astype(np.float32)
    T = len(bf)
    del bf

    etiq = pd.DataFrame()
    if ruta_roiset:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from leer_roiset import leer_roiset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            medidas = leer_roiset(ruta_roiset)
        etiq = proyectar_lineas(medidas, canales, info['periodo'], MARGEN_SUP)
        mapa = etiq.groupby('canal').canal_usuario.agg(lambda s: ','.join(map(str, sorted(s.unique()))))
        canales['canal_usuario'] = canales.canal.map(mapa).fillna('')
        etiq.to_csv(salida / f'{base}_kimo_etiquetas.csv', index=False)
        if len(etiq) < len(medidas):
            warnings.warn(f'{len(medidas) - len(etiq)} measurements do not fall in any detected channel')
    else:
        canales['canal_usuario'] = ''

    horas = _horas(etiquetas)
    np.savez_compressed(
        salida / f'{base}_kimogramas.npz', BF=K, BF_min=K_min, BF_desv=K_sd, brillo_BF=brillo,
        tiempo_min=((horas - horas.min()).dt.total_seconds() / 60).to_numpy(),
        etiquetas=np.array(etiquetas), canales=canales.to_records(index=False),
        margen_sup=MARGEN_SUP, ancho=ANCHO, periodo=info['periodo'], stack=str(ruta_bf.name))
    nombres = [f'canal {c:02d}' + (f' (usuario {u})' if u else '') for c, u in zip(canales.canal, canales.canal_usuario)]
    tifffile.imwrite(salida / f'{base}_kimo_BF.tif', np.stack([normalizar(k) for k in K]), imagej=True,
                     metadata={'Labels': nombres, 'axes': 'ZYX',
                               'Overlays': _superposicion(canales, etiq, largo, T)})
    canales.to_csv(salida / f'{base}_canales.csv', index=False)
    if figura:
        figura_kimogramas(K, canales, etiq, salida / f'{base}_kimogramas.png', titulo=ruta_bf.name)
    print(f'{ruta_bf.name}: {len(canales)} channels (period {info["periodo"]} px), kymographs of '
          f'{largo} px × {T} slices' + (f', {len(etiq)} measurements projected' if len(etiq) else ''))
    return canales, etiq


def figura_kimogramas(K, canales, etiq, ruta, titulo='', por_fila=8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(K)
    nfil = int(np.ceil(n / por_fila))
    fig, axs = plt.subplots(nfil, por_fila, figsize=(2.3 * por_fila, 3.3 * nfil), squeeze=False)
    for i, ax in enumerate(axs.flat):
        if i >= n:
            ax.axis('off')
            continue
        k = normalizar(K[i])
        ax.imshow(k, cmap='gray', aspect='auto', interpolation='nearest')
        c = canales.iloc[i]
        ax.axhline(c.fila_fondo, color='#eda100', lw=0.6, ls=':')
        ax.axhline(c.fila_salida, color='#eda100', lw=0.8, ls='--')
        if len(etiq):
            e = etiq[etiq.canal == c.canal]
            ax.plot(e.slice - 1, e.fila_inf, '.', color='#e34948', ms=2)
            ax.plot(e.slice - 1, e.fila_sup, '.', color='#1baf7a', ms=2)
        u = c.get('canal_usuario', '')
        ax.set_title(f'channel {c.canal}' + (f'  (user {u})' if u else ''), fontsize=8, loc='left')
        ax.set_xlim(-0.5, k.shape[1] - 0.5); ax.set_ylim(k.shape[0] - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{titulo}  ·  red/green: poles of your lines · dashed: channel exit', fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=90)
    plt.close(fig)


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    p = argparse.ArgumentParser(description='Per-channel kymographs of an aligned stack.')
    p.add_argument('stacks', nargs='+', help='bright-field stacks (-BF or _alineado)')
    p.add_argument('--salida', help='output folder (default <folder>/Kimogramas)')
    a = p.parse_args()
    for st in a.stacks:
        generar_kimogramas(st, salida=a.salida)
