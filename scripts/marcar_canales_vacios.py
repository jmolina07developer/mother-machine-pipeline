"""Apply detectar_celulas.canal_vacio to existing lineages (without redoing the tracking).
Usage: python marcar_canales_vacios.py <carpeta Stacks> [...]   -> rewrites _linaje.csv and flags _linaje_ciclos.csv"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectar_celulas as DC
for st in sys.argv[1:]:
    tot = 0; pos_cambiadas = []
    for f in sorted(Path(st, 'Canales').glob('*_canales_BF_linaje.csv')):
        base = f.name.replace('_canales_BF_linaje.csv', '')
        geo = pd.read_csv(f.with_name(f'{base}_canales_BF_geometria.csv')).set_index('canal')
        F = pd.read_csv(f); K = pd.read_csv(f.with_name(f'{base}_canales_BF_linaje_ciclos.csv'))
        partes = []
        for c, g in F.groupby('canal'):
            Cc = K[K.canal == c].assign(final=lambda d: d.final.astype(str).map(lambda s: 'division' if s.startswith('se divide') else s))
            g2 = DC.canal_vacio(g.drop(columns=['canal_vacio'], errors='ignore'), Cc, float(geo.fila_fondo[c]), float(geo.fila_salida[c]))
            partes.append(g2)
        F2 = pd.concat(partes).sort_index()
        vac = F2[F2.canal_vacio].groupby(['canal', 'celula']).size()
        K['canal_vacio'] = [(c, cel) in vac.index for c, cel in zip(K.canal, K.celula)]
        F2.to_csv(f, index=False); K.to_csv(f.with_name(f'{base}_canales_BF_linaje_ciclos.csv'), index=False)
        if len(vac): pos_cambiadas.append(base); tot += len(vac)
    print(f'{st}: {tot} "cells" that were empty channels in {len(pos_cambiadas)} positions')
    Path(st, 'canales_vacios_posiciones.txt').write_text('\n'.join(pos_cambiadas))
