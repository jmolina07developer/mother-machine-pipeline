"""
Runs the whole pipeline for one experiment, the same steps as pipeline_completo.ipynb, from a settings file.
The graphical interface (interfaz.py) starts it in the background; it can also be used from the terminal:

    python ejecutar_pipeline.py settings.json

settings.json (every key is optional except carpeta_imagenes):
    {"carpeta_imagenes": "/path/to/MyExperiment/Images",
     "posiciones": null,                 # null = all, or ["01", "02"]
     "solo_bf": null,                    # null = detect; true = bright field only
     "fluo": ["PI"],                     # fluorescence names, from the brightest to the dimmest
     "rehacer": false,                   # redo the steps even if their results exist
     "rehacer_stacks": [],               # positions whose stacks must be redone (also those listed in
                                         # Stacks/interfaz/rehacer_stacks.txt, written when images are discarded)
     "pasos": {"stacks": true, "canales": true, "celulas": true, "cajas": true,
               "resumen": true, "excel": true, "fluorescencia": true},
     "zona": "numeros",                  # alignment region: "numeros" (chip numbers) or "imagen" (whole image)
     "margen": 80}                       # maximum shift searched (px)

Progress goes to <experiment>/Stacks/interfaz/estado.json (step, position, i of n, state) and the messages to
standard output. Creating <experiment>/Stacks/interfaz/parar stops the run after the current position.
"""
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASOS = ['stacks', 'canales', 'celulas', 'cajas', 'resumen', 'excel', 'fluorescencia']
NOMBRE_PASO = {'inventario': 'Reading the images', 'areas': 'Checking that every image belongs to its position',
               'stacks': '1. Aligned stacks', 'canales': '2. Straightened channels', 'celulas': '3. Cells and lineage',
               'cajas': '4. Stacks with boxes', 'resumen': '5. Summary', 'excel': "6. Mothers' Excel files",
               'fluorescencia': '7. Fluorescence per cell'}
ZONA_IMAGEN = (0, 100000)          # the whole image as alignment region


class Parada(Exception):
    pass


class Estado:
    """Writes the progress to estado.json (read by the interface)."""
    def __init__(self, carpeta):
        self.carpeta = Path(carpeta); self.carpeta.mkdir(parents=True, exist_ok=True)
        self.f = self.carpeta / 'estado.json'; self.parar = self.carpeta / 'parar'
        if self.parar.exists():
            self.parar.unlink()
        self.d = dict(estado='en marcha', inicio=time.time(), paso='', nombre_paso='', pos='', i=0, n=0,
                      mensaje='', hechos=[])

    def poner(self, **k):
        self.d.update(k); self.d['actualizado'] = time.time()
        tmp = self.f.with_suffix('.tmp'); tmp.write_text(json.dumps(self.d)); tmp.replace(self.f)

    def comprobar(self):
        if self.parar.exists():
            raise Parada()


def ejecutar(cfg):
    import numpy as np  # noqa: F401
    import pandas as pd
    import generar_stacks as GS, canales_enderezados as CE, detectar_celulas as DC, stack_linaje as SL
    import excel_madres as EM, fluorescencia_linaje as FL

    imagenes = Path(cfg['carpeta_imagenes']).expanduser().resolve()
    stacks = imagenes.parent / 'Stacks'; canales = stacks / 'Canales'; linaje = stacks / 'Linaje'
    est = Estado(stacks / 'interfaz')
    pasos = {p: bool(cfg.get('pasos', {}).get(p, True)) for p in PASOS}
    posiciones = cfg.get('posiciones') or None
    fluos = list(cfg.get('fluo') or ['PI']); fluos = [f.strip() for f in fluos if f.strip()] or ['PI']
    rehacer = bool(cfg.get('rehacer', False)); rehacer_stacks = set(cfg.get('rehacer_stacks') or [])
    f_rehacer = stacks / 'interfaz' / 'rehacer_stacks.txt'     # positions whose images were discarded after their stack
    if f_rehacer.exists():
        rehacer_stacks |= set(f_rehacer.read_text().split())
    f_forzar = stacks / 'interfaz' / 'rehacer_pasos.json'      # step and position interrupted by "Stop now": redone
    forzar = json.loads(f_forzar.read_text()) if f_forzar.exists() else {}

    def hecho_forzado(nombre, p):
        if p in forzar.get(nombre, []):
            forzar[nombre].remove(p); f_forzar.write_text(json.dumps(forzar))
    zona = ZONA_IMAGEN if cfg.get('zona') == 'imagen' else GS.ZONA_NUMEROS
    margen = int(cfg.get('margen') or GS.MARGEN_STACKS)
    print(f'Images : {imagenes}\nResults: {stacks}\nSteps  : ' + ', '.join(p for p in PASOS if pasos[p]), flush=True)
    if not imagenes.exists():
        raise SystemExit(f'The folder {imagenes} does not exist.')

    def paso(nombre, i=0, n=0, pos=''):
        est.comprobar()
        est.poner(paso=nombre, nombre_paso=NOMBRE_PASO.get(nombre, nombre), i=i, n=n, pos=pos)

    def bf_hechos():
        return {p.name.split('-')[0]: p for p in sorted(stacks.glob('*-BF.tif'))
                if not posiciones or p.name.split('-')[0] in posiciones}

    # 1. stacks
    if pasos['stacks']:
        paso('inventario')
        inv, umbral = GS.inventario(imagenes, fluos)
        inv = GS.aplicar_canales(inv, stacks)
        inv = GS.quitar_descartadas(inv, stacks)
        todas = sorted(inv.pos.unique()) if not posiciones else [p for p in posiciones if p in set(inv.pos)]
        hechos = bf_hechos()
        faltan = [p for p in todas if rehacer or p in rehacer_stacks or p not in hechos]
        if faltan:
            paso('areas')
            inv, _ = GS.corregir_areas(imagenes, inv, stacks)
            solo_bf = cfg.get('solo_bf')
            if solo_bf is None:
                solo_bf = not GS.hay_pi(inv)
            if solo_bf:
                inv['canal'] = 'BF'
            print(f'{len(inv)} images | ' + ('bright field only' if solo_bf else
                  f'BF: {(inv.canal == "BF").sum()}, fluorescence: {(inv.canal != "BF").sum()} ({", ".join(fluos)})'),
                  flush=True)
            resumenes = []
            for i, p in enumerate(faltan):
                paso('stacks', i, len(faltan), p)
                inv_pos = inv[inv.pos == p]
                if len(inv_pos) < GS.POS_MIN_IMAGENES:
                    print(f'Position {p}: only {len(inv_pos)} images, ignored', flush=True); continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    for inv_s in (GS.separar_series(imagenes, inv_pos) if solo_bf else [inv_pos]):
                        if len(inv_s) < GS.POS_MIN_IMAGENES: continue
                        resumenes.append(GS.generar_posicion(imagenes, inv_s, stacks, True, zona, margen, True, solo_bf,
                                                             fluos[0] if len(fluos) == 1 else fluos))
                res = pd.DataFrame(resumenes[-1:])
                f = stacks / 'resumen_stacks.csv'
                if f.exists():
                    viejo = pd.read_csv(f, dtype={'pos': str})
                    res = pd.concat([viejo[~viejo.pos.isin(res.pos)], res]).sort_values('pos')
                res.to_csv(f, index=False)
                if f_rehacer.exists():                       # done: no longer pending
                    resto = [x for x in f_rehacer.read_text().split() if x != p]
                    f_rehacer.write_text(' '.join(resto))
        else:
            print('Stacks already done.', flush=True)
    BF = bf_hechos()
    if not BF:
        raise SystemExit('There are no stacks yet (step 1).')

    def por_posicion(nombre, hacer):
        lista = list(BF.items())
        for i, (p, r) in enumerate(lista):
            paso(nombre, i, len(lista), p)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if hacer(p, r, p in forzar.get(nombre, [])):
                    print(f'   {p}: {time.time() - t0:.0f} s', flush=True)
            hecho_forzado(nombre, p)

    # 2. channels
    if pasos['canales']:
        def h(p, r, forzado=False):
            ce = canales / f"{r.stem.replace('-BF', '')}_canales_BF.tif"
            if ce.exists() and ce.stat().st_mtime > r.stat().st_mtime and not (rehacer or forzado):
                return False
            CE.canales_enderezados(r); return True
        por_posicion('canales', h)

    # 3. cells and lineage
    if pasos['celulas']:
        modelo = None
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            import joblib
            modelo = joblib.load(DC.MODELO); print('onnxruntime is missing: using the tree-based detector', flush=True)
        dc = Path(DC.__file__).parent
        t_prog = max((dc / f).stat().st_mtime for f in ('detectar_celulas.py', 'linaje.py', 'modelos/detector_centros.onnx')
                     if (dc / f).exists())

        def h(p, r, forzado=False):
            img = canales / f"{r.stem.replace('-BF', '')}_canales_BF.tif"
            if not img.exists():
                print(f'{p}: straightened channels missing (step 2)', flush=True); return False
            lin = img.with_name(img.stem + '_linaje.csv'); zp = img.with_name(img.stem + '_cajas.zip')
            t_cajas = zp.stat().st_mtime if zp.exists() else 0
            if lin.exists() and lin.stat().st_mtime > max(img.stat().st_mtime, t_prog, t_cajas) and not (rehacer or forzado):
                return False
            DC.predecir(img, modelo); return True
        por_posicion('celulas', h)

    # 4. stacks with boxes
    if pasos['cajas']:
        def h(p, r, forzado=False):
            dest = linaje / f'{r.stem}_linaje.tif'
            lin = canales / f"{r.stem.replace('-BF', '')}_canales_BF_linaje.csv"
            if not lin.exists() or (dest.exists() and dest.stat().st_mtime > max(lin.stat().st_mtime,
                                                                                  Path(SL.__file__).stat().st_mtime) and not (rehacer or forzado)):
                return False
            SL.stack_linaje(r, canales=canales, salida=linaje); return True
        por_posicion('cajas', h)

    # 5. summary (all cells of all positions)
    if pasos['resumen']:
        paso('resumen')
        Ks = []
        for p, r in BF.items():
            f = canales / f"{r.stem.replace('-BF', '')}_canales_BF_linaje_ciclos.csv"
            if f.exists():
                Ks.append(pd.read_csv(f).assign(pos=p))
        if Ks:
            pd.concat(Ks).to_csv(stacks / 'linaje_ciclos_todas.csv', index=False)
            print(f'All cells of all positions: {stacks / "linaje_ciclos_todas.csv"}', flush=True)

    # 6. Excel
    if pasos['excel']:
        def h(p, r, forzado=False):
            base = r.stem.replace('-BF', '')
            lin = canales / f'{base}_canales_BF_linaje.csv'; dest = stacks / 'Excel' / f'{base}_longitud_madres.xlsx'
            if not lin.exists() or (dest.exists() and dest.stat().st_mtime > lin.stat().st_mtime and not (rehacer or forzado)):
                return False
            EM.excel_madres(r, canales=canales); return True
        por_posicion('excel', h)

    # 7. fluorescence
    if pasos['fluorescencia']:
        def h(p, r, forzado=False):
            base = r.stem.replace('-BF', ''); lin = canales / f'{base}_canales_BF_linaje.csv'; hecho = False
            for f in fluos:
                st = r.with_name(f'{base}-{f}.tif'); dest = canales / f'{base}_canales_BF_linaje_{f}.csv'
                if not st.exists() or not lin.exists():
                    continue
                if dest.exists() and dest.stat().st_mtime > lin.stat().st_mtime and not (rehacer or forzado):
                    continue
                FL.fluorescencia_linaje(r, fluo=f, canales=canales); hecho = True
            return hecho
        por_posicion('fluorescencia', h)
    return est


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cfg = json.loads(Path(sys.argv[1]).read_text())
    stacks = Path(cfg['carpeta_imagenes']).expanduser().resolve().parent / 'Stacks'
    est = None
    try:
        est = ejecutar(cfg)
        est.poner(estado='terminado', fin=time.time(), mensaje='Finished')
        print('\nFINISHED', flush=True)
    except Parada:
        print('\nStopped by the user.', flush=True)
        _final(stacks, 'parado', 'Stopped by the user')
    except BaseException as e:  # noqa: B902
        traceback.print_exc()
        sys.stdout.flush()
        _final(stacks, 'error', f'{type(e).__name__}: {e}')
        raise SystemExit(1)


def _final(stacks, estado, mensaje):
    f = stacks / 'interfaz' / 'estado.json'
    d = json.loads(f.read_text()) if f.exists() else {}
    d.update(estado=estado, mensaje=mensaje, fin=time.time(), actualizado=time.time())
    f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(d))


if __name__ == '__main__':
    main()
