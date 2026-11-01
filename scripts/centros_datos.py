"""
Data for the centre-based detector: crops of the annotated channels (uint8, all images, so that the
network can be given the neighbouring images) and, for each annotated image, its boxes, the erased rows
and the per-row labels (the same ones the current detector uses).
Usage: centros_datos.py carpeta_con_zips salida.pkl
"""
import sys, glob, pickle, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import detectar_celulas as DC
from leer_cajas import leer_cajas, fotos_malas, malas_del_canal, leer_borradas

from centros_comun import ANCHO, LARGO, recorte_fijo


def construir(carpeta):
    canales, fotos = {}, []
    for rz in sorted(glob.glob(f'{carpeta}/*_cajas.zip')):
        rz = Path(rz); base = rz.name.replace('_canales_BF_cajas.zip', '')
        ruta_img = rz.with_name(base + '_canales_BF.tif')
        cajas, vacias, anot = leer_cajas(rz)
        if not len(anot): continue
        img, geo = DC.abrir(ruta_img)
        uf = DC.ultima_foto(ruta_img)
        if uf is not None:
            anot = anot[anot.foto <= uf]; cajas = cajas[cajas.foto <= uf]
        malas = fotos_malas(rz)
        if len(malas):
            es_mala = np.array([f in set(malas_del_canal(malas, c)) for c, f in zip(anot.canal, anot.foto)], bool)
            anot = anot[~es_mala]
        test = DC.conjunto_prueba(rz); borr = leer_borradas(rz)
        for c in sorted(anot.canal.unique()):
            k = int(np.where(geo.canal.values == c)[0][0]); g = geo.iloc[k]
            T = int(g.fotos) if uf is None else min(int(uf), int(g.fotos))
            T3 = DC.tiles(img, k, int(g.ancho), int(g.sep), T).astype(np.uint8)
            L = T3.shape[1]
            canales[(base, int(c))] = dict(tiles=recorte_fijo(T3), L=L, fila_fondo=float(g.fila_fondo),
                                           fila_salida=float(g.fila_salida), prueba=int(c) in test)
            for f in sorted(anot[anot.canal == c].foto.unique()):
                cf = cajas[(cajas.canal == c) & (cajas.foto == f)].sort_values('fila_sup')
                bb = [(r.y0, r.y1) for r in borr[(borr.canal == c) & (borr.foto == f)].itertuples()]
                y = DC.etiquetas(cf, L, bb)
                w = np.where(DC.filas_borradas(bb, L), DC.PESO_BORRADAS, 1.0)
                abajo = y < 0; y[abajo] = 0; w[abajo] = DC.PESO_ABAJO
                fotos.append(dict(base=base, canal=int(c), foto=int(f), prueba=int(c) in test,
                                  cajas=[(float(r.fila_sup), float(r.fila_inf)) for r in cf.itertuples()],
                                  borradas=bb, y=y, w=w.astype(np.float32), L=L))
        print(f'{base}: {sum(1 for f in fotos if f["base"] == base)} annotated images', flush=True)
    return canales, fotos


if __name__ == '__main__':
    canales, fotos = construir(sys.argv[1])
    pickle.dump(dict(canales=canales, fotos=fotos, ANCHO=ANCHO, LARGO=LARGO), open(sys.argv[2], 'wb'), protocol=4)
    print(len(canales), 'channels,', len(fotos), 'images,', sum(f['prueba'] for f in fotos), 'test')
