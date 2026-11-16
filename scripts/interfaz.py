"""
Graphical interface of the mother machine pipeline (runs in the web browser, on your own computer).

Start it with the launcher (start_windows.bat / start_mac.command) or from the scripts folder:
    streamlit run interfaz.py

Everything it does can also be done with the programs or the notebook; it writes the same files:
  - Stacks/canal_imagenes.csv         channel of each image set by hand (when two fluorescences are equally bright)
  - Stacks/imagenes_descartadas.csv   images left out
  - Stacks/interfaz/                  settings, progress and messages of the last run
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import generar_stacks as GS  # noqa: E402

st.set_page_config(page_title='Mother machine pipeline', page_icon='🔬', layout='wide')
RECIENTES = Path.home() / '.mother_machine_interfaz.json'
PASOS = [('stacks', '1. Aligned stacks'), ('canales', '2. Channels'), ('celulas', '3. Cells and lineage'),
         ('cajas', '4. Boxes on the stacks'), ('resumen', '5. Summary table'), ('excel', "6. Mothers' Excel"),
         ('fluorescencia', '7. Fluorescence')]


# ---------------------------------------------------------------------------------------------- helpers
def leer_json(f, defecto=None):
    try:
        return json.loads(Path(f).read_text())
    except Exception:
        return {} if defecto is None else defecto


def escribir_json(f, d):
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    Path(f).write_text(json.dumps(d, indent=1))


def abrir(ruta):
    """Opens a file or folder with the default application of the computer."""
    ruta = str(ruta)
    try:
        if sys.platform.startswith('win'):
            os.startfile(ruta)  # noqa: S606
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', ruta])
        else:
            subprocess.Popen(['xdg-open', ruta])
    except Exception as e:
        st.error(f'Could not open {ruta}: {e}')


def elegir_carpeta():
    """Folder dialog of the operating system (in a separate process, so it does not block the interface)."""
    codigo = ('import tkinter as tk; from tkinter import filedialog; r = tk.Tk(); r.withdraw(); '
              'r.attributes("-topmost", True); print(filedialog.askdirectory(title="Folder with the images of the experiment"))')
    try:
        out = subprocess.run([sys.executable, '-c', codigo], capture_output=True, text=True, timeout=600)
        return out.stdout.strip() or None
    except Exception:
        return None


def vivo(pid):
    if not pid:
        return False
    if sys.platform.startswith('win'):
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code)); ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                          # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    try:                                                                  # a finished child that was not collected
        r, _ = os.waitpid(int(pid), os.WNOHANG)
        return r == 0
    except ChildProcessError:
        return True


def matar(pid):
    if sys.platform.startswith('win'):
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except Exception:
                pass


@st.cache_data(max_entries=4000, show_spinner=False)
def miniatura(ruta, mtime, ancho=300, pagina=None, rgb=False):
    """Small 8-bit version of an image (or of one page of a stack) for display."""
    import tifffile
    from PIL import Image
    with tifffile.TiffFile(ruta) as tf:
        a = tf.pages[pagina or 0].asarray()
    if rgb and a.ndim == 3:
        im = Image.fromarray(a.astype(np.uint8))
    else:
        a = a.astype(np.float32); lo, hi = np.percentile(a[::4, ::4], [0.5, 99.8])
        im = Image.fromarray(np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8))
    h = int(im.height * ancho / im.width)
    return np.asarray(im.resize((ancho, h)))


def n_paginas(ruta):
    import tifffile
    with tifffile.TiffFile(ruta) as tf:
        return len(tf.pages)


@st.cache_data(show_spinner=False)
def inventario(carpeta, clave, fluos):
    inv, umbral = GS.inventario(Path(carpeta), list(fluos))
    return inv, umbral


def clave_carpeta(carpeta):
    fs = [f for f in Path(carpeta).iterdir() if GS.PATRON.match(f.name)]
    return (len(fs), max((f.stat().st_mtime for f in fs), default=0))


def duenios_fluo(inv):
    """For each bright-field image, the fluorescence images that belong to it (same position, < 60 s)."""
    bf = inv[inv.canal == 'BF']; fl = inv[inv.canal != 'BF']; d = {}
    for p, g in fl.groupby('pos'):
        b = bf[bf.pos == p]
        if b.empty: continue
        for _, f in g.iterrows():
            dt = (b.hora - f.hora).dt.total_seconds().abs()
            if dt.min() <= GS.MAX_SEGUNDOS:
                d.setdefault(b.archivo[dt.idxmin()], []).append(f.archivo)
    return d


def leer_descartadas(stacks):
    f = stacks / 'imagenes_descartadas.csv'
    if f.exists():
        t = pd.read_csv(f, dtype=str).fillna('')
        return dict(zip(t.archivo, t.get('motivo', [''] * len(t))))
    return {}


def guardar_descartadas(stacks, d, inv=None):
    """Writes imagenes_descartadas.csv; positions whose stack already exists are marked to be redone."""
    viejo = leer_descartadas(stacks)
    stacks.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(archivo=list(d), motivo=list(d.values()))).to_csv(stacks / 'imagenes_descartadas.csv', index=False)
    cambiados = set(viejo) ^ set(d)
    pos = sorted({a.split('_')[0] for a in cambiados})
    rehacer = [p for p in pos if list(stacks.glob(f'{p}-*-BF.tif'))]
    if rehacer:
        f = stacks / 'interfaz' / 'rehacer_stacks.txt'; f.parent.mkdir(parents=True, exist_ok=True)
        actual = set(f.read_text().split()) if f.exists() else set()
        f.write_text(' '.join(sorted(actual | set(rehacer))))
    return rehacer


def recordar(carpeta):
    d = leer_json(RECIENTES, {'recientes': []}); r = [c for c in d.get('recientes', []) if c != carpeta]
    d['recientes'] = [carpeta] + r[:9]
    try:
        escribir_json(RECIENTES, d)
    except Exception:
        pass


# ---------------------------------------------------------------------------------------------- sidebar
st.sidebar.title('🔬 Mother machine')
if 'carpeta' not in st.session_state:
    rec = leer_json(RECIENTES, {'recientes': []}).get('recientes', [])
    st.session_state.carpeta = rec[0] if rec else ''
if st.sidebar.button('📂 Choose the images folder…', help='Choose the folder with the microscope images of one experiment',
                     width='stretch'):
    r = elegir_carpeta()
    if r:
        st.session_state.carpeta = r
        st.session_state.pop('campo_carpeta', None)
carpeta = st.sidebar.text_input('Images folder of the experiment', value=st.session_state.carpeta, key='campo_carpeta',
                                help='The folder with all the .tif images of one experiment (all positions and times). '
                                     'The results go to the folder Stacks next to it.')
st.session_state.carpeta = carpeta
rec = leer_json(RECIENTES, {'recientes': []}).get('recientes', [])
if rec:
    elegida = st.sidebar.selectbox('Recent experiments', [''] + rec, index=0, format_func=lambda x: x or '—')
    if elegida and elegida != carpeta:
        st.session_state.carpeta = elegida; st.session_state.pop('campo_carpeta', None); st.rerun()

IMAGENES = Path(carpeta).expanduser() if carpeta else None
if not IMAGENES or not IMAGENES.is_dir():
    st.title('Mother machine pipeline')
    st.info('Choose the folder with the images of an experiment in the panel on the left (**Browse…**). '
            'The images must be named `<position>_<yymmdd>_<hhmmss>_anything.tif`, e.g. '
            '`04_260925_103958_Regrowth_BF_00300_.tif`; the results are written to the folder `Stacks` next to it.')
    if carpeta:
        st.error(f'The folder {carpeta} does not exist.')
    st.stop()
recordar(str(IMAGENES))
STACKS = IMAGENES.parent / 'Stacks'; CANALES = STACKS / 'Canales'; LINAJE = STACKS / 'Linaje'; INTERFAZ = STACKS / 'interfaz'
AJUSTES = INTERFAZ / 'ajustes_interfaz.json'
aj = leer_json(AJUSTES, {})
st.sidebar.caption(f'Results: `{STACKS}`')
if st.sidebar.button('Open the results folder', disabled=not STACKS.exists()):
    abrir(STACKS)
st.sidebar.divider()
st.sidebar.subheader('Fluorescence')
solo_bf = st.sidebar.checkbox('Bright field only (no fluorescence images)', value=aj.get('solo_bf', False))
fluo_txt = st.sidebar.text_input('Fluorescence names, from the brightest to the dimmest', value=aj.get('fluo', 'PI'),
                                 disabled=solo_bf, help='e.g. "PI", or "GFP, PI" when there are two. They are told '
                                 'apart by brightness, or by time in the tab Images if they are equally bright.')
FLUOS = [f.strip() for f in fluo_txt.replace(';', ',').split(',') if f.strip()] or ['PI']
if aj.get('fluo') != fluo_txt or aj.get('solo_bf') != solo_bf:
    aj.update(fluo=fluo_txt, solo_bf=solo_bf); escribir_json(AJUSTES, aj)

tab_img, tab_run, tab_ali, tab_res, tab_ayuda = st.tabs(['1 · Images', '2 · Run', '3 · Check alignment', '4 · Results', 'Help'])

# ---------------------------------------------------------------------------------------------- 1. images
with tab_img:
    st.header('Images of the experiment')
    with st.spinner('Reading the images (the first time takes ~1 s per 100 images)…'):
        try:
            inv, umbral = inventario(str(IMAGENES), clave_carpeta(IMAGENES), tuple(FLUOS))
        except FileNotFoundError:
            inv = pd.DataFrame()
    todas = sorted(p.name for p in IMAGENES.glob('*.tif*'))
    if inv.empty:
        st.error(f'There are {len(todas)} .tif files but none has the expected name '
                 '(`<position>_<yymmdd>_<hhmmss>_anything.tif`). Examples: ' + ', '.join(todas[:3]))
        st.stop()
    if len(inv) < len(todas):
        st.warning(f'{len(todas) - len(inv)} of {len(todas)} files do not have the expected name and are ignored.')
    inv_auto = inv.copy()
    if solo_bf:
        inv['canal'] = 'BF'
    inv = GS.aplicar_canales(inv, STACKS) if (STACKS / 'canal_imagenes.csv').exists() and not solo_bf else inv
    desc = leer_descartadas(STACKS)
    inv['descartada'] = inv.archivo.isin(desc)

    tabla = inv[~inv.descartada].groupby('pos').agg(imagenes=('archivo', 'size'), primera=('hora', 'min'), ultima=('hora', 'max'))
    for c in ['BF'] + ([] if solo_bf else FLUOS):
        tabla[c] = inv[~inv.descartada & (inv.canal == c)].groupby('pos').size().reindex(tabla.index).fillna(0).astype(int)
    tabla['discarded'] = inv[inv.descartada].groupby('pos').size().reindex(tabla.index).fillna(0).astype(int)
    tabla['primera'] = tabla.primera.dt.strftime('%H:%M'); tabla['ultima'] = tabla.ultima.dt.strftime('%H:%M')
    a, b, c = st.columns(3)
    a.metric('Images', len(inv)); b.metric('Positions', inv.pos.nunique())
    c.metric('Discarded', int(inv.descartada.sum()))
    st.dataframe(tabla.rename(columns={'imagenes': 'images', 'primera': 'first', 'ultima': 'last'}), width='stretch',
                 height=min(38 * (len(tabla) + 1), 400))
    if (inv.groupby('pos').size() < GS.POS_MIN_IMAGENES).any():
        st.caption(f'Positions with fewer than {GS.POS_MIN_IMAGENES} images are ignored (tests, stray images).')

    if not solo_bf:
        st.subheader('Bright field and fluorescence')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.2))
        colores = {'BF': '#555555'} | {f: c for f, c in zip(FLUOS, ['#2ca02c', '#d62728', '#1f77b4', '#9467bd'])}
        for cn, g in inv.groupby('canal'):
            ax.scatter(g.hora, g.mediana, s=6, label=f'{cn} ({len(g)})', color=colores.get(cn, '#ff7f0e'))
        ax.set_yscale('log'); ax.set_ylabel('image brightness (median)'); ax.legend(fontsize=8, markerscale=2)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); ax.set_xlabel('time of the image')
        ax.grid(alpha=0.3); st.pyplot(fig, width='stretch'); plt.close(fig)
        st.caption('Each dot is one image. Bright-field images are the brightest; the fluorescence ones are told apart '
                   'by brightness. If two fluorescences overlap (e.g. GFP and PI), assign them by time below.')
        with st.expander('Assign the fluorescence by time (when two fluorescences are equally bright)',
                         expanded=(STACKS / 'canal_imagenes.csv').exists()):
            reglas0 = pd.DataFrame(aj.get('reglas', []), columns=['fluorescence', 'from', 'to']).astype(str)
            st.write('Fluorescence images taken between these times (HH:MM, both included) belong to the fluorescence '
                     'of their row; add one row per round (click the empty row), e.g. PI from 16:36 to 16:46. Look at the '
                     'times in the plot above or in the images below.')
            reglas = st.data_editor(reglas0, num_rows='dynamic', key='reglas', width='stretch', column_config={
                'fluorescence': st.column_config.SelectboxColumn(options=FLUOS, required=True),
                'from': st.column_config.TextColumn(help='HH:MM'), 'to': st.column_config.TextColumn(help='HH:MM')})
            c1, c2 = st.columns(2)
            ultima = c1.selectbox('The last fluorescence image of each position is', ['(no rule)'] + FLUOS,
                                  index=(['(no rule)'] + FLUOS).index(aj.get('ultima', '(no rule)'))
                                  if aj.get('ultima', '(no rule)') in ['(no rule)'] + FLUOS else 0)
            resto = c2.selectbox('All the other fluorescence images are', FLUOS,
                                 index=FLUOS.index(aj['resto']) if aj.get('resto') in FLUOS else 0)
            nuevo = inv_auto.copy()
            esfl = nuevo.canal != 'BF'
            nuevo.loc[esfl, 'canal'] = resto
            hm = nuevo.hora.dt.strftime('%H:%M')
            malas = []
            for _, r in reglas.dropna().iterrows():
                desde, hasta = str(r['from']).strip().zfill(5), str(r['to']).strip().zfill(5)
                if len(desde) != 5 or len(hasta) != 5 or desde[2] != ':' or hasta[2] != ':':
                    malas.append(f'{desde}–{hasta}'); continue
                nuevo.loc[esfl & (hm >= desde) & (hm <= hasta), 'canal'] = r['fluorescence']
            if ultima != '(no rule)':
                ult = nuevo[esfl].groupby('pos').hora.idxmax()
                nuevo.loc[ult.values, 'canal'] = ultima
            if malas:
                st.error('Write the times as HH:MM: ' + ', '.join(malas))
            prev = nuevo[esfl].groupby(['pos', 'canal']).size().unstack(fill_value=0)
            prev.index.name = 'position'; prev.columns.name = 'channel'
            st.write('Fluorescence images per position with these rules:')
            st.dataframe(prev.T, width='stretch')
            c1, c2 = st.columns(2)
            if c1.button('Save this assignment', type='primary'):
                STACKS.mkdir(parents=True, exist_ok=True)
                nuevo[['archivo', 'canal']].to_csv(STACKS / 'canal_imagenes.csv', index=False)
                aj.update(reglas=reglas.dropna().to_dict('records'), ultima=ultima, resto=resto); escribir_json(AJUSTES, aj)
                st.success('Saved in Stacks/canal_imagenes.csv. It is used from now on when the stacks are made '
                           '(redo step 1 if they were already made).'); st.rerun()
            if c2.button('Go back to the assignment by brightness', disabled=not (STACKS / 'canal_imagenes.csv').exists()):
                (STACKS / 'canal_imagenes.csv').unlink(); st.rerun()

    st.subheader('Repeated images')
    bf = inv[(inv.canal == 'BF')].sort_values(['pos', 'hora'])
    pares = []
    for p, g in bf.groupby('pos'):
        g = g.reset_index(drop=True); dt = g.hora.diff().dt.total_seconds()
        for i in np.where(dt < 240)[0]:
            pares.append((p, g.loc[i - 1], g.loc[i]))
    duenio = duenios_fluo(inv)
    if not pares:
        st.write('No position has two bright-field images taken less than 4 minutes apart.')
    else:
        st.write(f'{len(pares)} times a position has two bright-field images less than 4 minutes apart: a retaken '
                 'photo or a photo of another field of the chip. Tick the one to leave out (its fluorescence goes with it).')
        nuevo_desc = dict(desc)
        for p, a1, a2 in pares:
            cols = st.columns([1, 2, 2])
            cols[0].markdown(f'**Position {p}**  \n{a1.hora:%H:%M:%S} / {a2.hora:%H:%M:%S}')
            for col, x in zip(cols[1:], (a1, a2)):
                col.image(miniatura(str(IMAGENES / x.archivo), (IMAGENES / x.archivo).stat().st_mtime, 360),
                          caption=f'{x.archivo}')
                fuera = col.checkbox('Leave out', value=x.archivo in desc, key=f'rep_{x.archivo}')
                for a in [x.archivo] + duenio.get(x.archivo, []):
                    if fuera: nuevo_desc.setdefault(a, 'repeated image (chosen in the interface)')
                    else: nuevo_desc.pop(a, None)
        if nuevo_desc != desc and st.button('Save the repeated images to leave out', type='primary'):
            red = guardar_descartadas(STACKS, nuevo_desc)
            st.success('Saved.' + (f' The stacks of positions {", ".join(red)} will be redone in step 1.' if red else ''))
            st.rerun()

    st.subheader('Look at the images of a position')
    pos_sel = st.selectbox('Position', sorted(inv.pos.unique()), key='pos_img')
    g = inv[(inv.pos == pos_sel) & (inv.canal == 'BF')].sort_values('hora').reset_index(drop=True)
    st.caption(f'{len(g)} bright-field images. Tick an image to leave it out (out of focus, another field of the chip, '
               'bubble…), together with its fluorescence image.')
    nuevo_desc = dict(desc); ncol = 6
    for i0 in range(0, len(g), ncol):
        cols = st.columns(ncol)
        for col, (_, x) in zip(cols, g.iloc[i0:i0 + ncol].iterrows()):
            f = IMAGENES / x.archivo
            col.image(miniatura(str(f), f.stat().st_mtime, 240), caption=f'{_ + 1}. {x.hora:%H:%M:%S}')
            fuera = col.checkbox('Leave out', value=x.archivo in desc, key=f'img_{x.archivo}')
            for a in [x.archivo] + duenio.get(x.archivo, []):
                if fuera: nuevo_desc.setdefault(a, 'left out in the interface')
                else: nuevo_desc.pop(a, None)
    if nuevo_desc != desc:
        if st.button(f'Save the images to leave out ({len(nuevo_desc)} in total)', type='primary', key='guardar_img'):
            red = guardar_descartadas(STACKS, nuevo_desc)
            st.success('Saved in Stacks/imagenes_descartadas.csv.' +
                       (f' The stacks of positions {", ".join(red)} will be redone in step 1.' if red else ''))
            st.rerun()

# ---------------------------------------------------------------------------------------------- 2. run
with tab_run:
    st.header('Run the pipeline')
    proc = leer_json(INTERFAZ / 'proceso.json', {})
    estado = leer_json(INTERFAZ / 'estado.json', {})
    activo = vivo(proc.get('pid')) and estado.get('estado', 'en marcha') == 'en marcha'
    posiciones = sorted(inv.pos.unique())
    c1, c2 = st.columns([2, 1])
    with c1:
        sel = st.multiselect('Positions (empty = all)', posiciones, default=aj.get('posiciones', []), disabled=activo)
        st.write('Steps:')
        cols = st.columns(3); pasos = {}
        for k, (clave, nombre) in enumerate(PASOS):
            pasos[clave] = cols[k % 3].checkbox(nombre, value=aj.get('pasos', {}).get(clave, True), disabled=activo,
                                                key=f'paso_{clave}')
    with c2:
        st.write('Alignment (step 1):')
        zona = st.radio('Region used to align', ['numeros', 'imagen'], disabled=activo,
                        index=1 if aj.get('zona') == 'imagen' else 0,
                        format_func=lambda z: 'Chip numbers (usual)' if z == 'numeros' else 'Whole image',
                        help='Use the whole image when the stage drifts a lot and the chip numbers leave the image.')
        margen = st.number_input('Maximum shift searched (px)', 20, 400, int(aj.get('margen', 80)),
                                 step=10, disabled=activo, help='80 is usually enough; 160 if the stage drifts a lot. '
                                 'Larger jumps are found anyway with a wide search.')
        if zona == 'imagen' and margen < 160:
            st.caption('With the whole image, a maximum shift of 160 px is usually better.')
        rehacer = st.checkbox('Redo the steps even if they were already done', value=False, disabled=activo)
    pend = (INTERFAZ / 'rehacer_stacks.txt').read_text().split() if (INTERFAZ / 'rehacer_stacks.txt').exists() else []
    if pend:
        st.info(f'The stack{"s" if len(pend) > 1 else ""} of position{"s" if len(pend) > 1 else ""} {", ".join(pend)} '
                'will be redone in step 1 (images were left out after they were made).')
        conzip = [p for p in pend if list(CANALES.glob(f'{p}-*_canales_BF_cajas.zip'))]
        if conzip:
            st.warning(f'Positions {", ".join(conzip)} have boxes corrected by hand: after redoing the stack the channels '
                       'may move, check them in Fiji.')
    b1, b2, b3 = st.columns(3)
    if b1.button('▶ Start', type='primary', disabled=activo, width='stretch'):
        if not any(pasos.values()):
            st.error('Choose at least one step.')
        else:
            cfg = dict(carpeta_imagenes=str(IMAGENES), posiciones=sel or None, solo_bf=True if solo_bf else None,
                       fluo=FLUOS, rehacer=rehacer, pasos=pasos, zona=zona, margen=int(margen))
            aj.update(posiciones=sel, pasos=pasos, zona=zona, margen=int(margen)); escribir_json(AJUSTES, aj)
            escribir_json(INTERFAZ / 'ajustes.json', cfg)
            if (INTERFAZ / 'parar').exists(): (INTERFAZ / 'parar').unlink()
            log = open(INTERFAZ / 'registro.txt', 'w', encoding='utf-8')
            log.write(f'Started {datetime.now():%Y-%m-%d %H:%M:%S}\n'); log.flush()
            kw = dict(creationflags=0x08000000) if sys.platform.startswith('win') else dict(start_new_session=True)
            env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
            p = subprocess.Popen([sys.executable, '-u', str(SCRIPTS / 'ejecutar_pipeline.py'), str(INTERFAZ / 'ajustes.json')],
                                 stdout=log, stderr=subprocess.STDOUT, cwd=str(SCRIPTS), env=env, **kw)
            escribir_json(INTERFAZ / 'proceso.json', dict(pid=p.pid, inicio=time.time()))
            escribir_json(INTERFAZ / 'estado.json', dict(estado='en marcha', inicio=time.time(), nombre_paso='Starting…'))
            time.sleep(1); st.rerun()
    if b2.button('⏸ Stop after this position', disabled=not activo, width='stretch'):
        (INTERFAZ / 'parar').write_text('stop'); st.toast('It will stop when the current position is finished.')
    if b3.button('⏹ Stop now', disabled=not activo, width='stretch'):
        matar(proc.get('pid')); time.sleep(1)
        e = leer_json(INTERFAZ / 'estado.json', {}); e.update(estado='parado', mensaje='Stopped by the user')
        if e.get('pos') and e.get('paso'):                  # what was half done is redone next time
            if e['paso'] == 'stacks':
                f = INTERFAZ / 'rehacer_stacks.txt'; actual = set(f.read_text().split()) if f.exists() else set()
                f.write_text(' '.join(sorted(actual | {e['pos']})))
            else:
                fz = leer_json(INTERFAZ / 'rehacer_pasos.json', {}); fz.setdefault(e['paso'], [])
                if e['pos'] not in fz[e['paso']]: fz[e['paso']].append(e['pos'])
                escribir_json(INTERFAZ / 'rehacer_pasos.json', fz)
        escribir_json(INTERFAZ / 'estado.json', e); st.rerun()

    @st.fragment(run_every=2)
    def progreso():
        proc = leer_json(INTERFAZ / 'proceso.json', {}); e = leer_json(INTERFAZ / 'estado.json', {})
        if not e:
            st.caption('Not run yet from the interface.'); return
        en_marcha = e.get('estado') == 'en marcha' and vivo(proc.get('pid'))
        if e.get('estado') == 'en marcha' and not en_marcha:
            e['estado'] = 'error'; e['mensaje'] = 'The process ended unexpectedly (see the messages below).'
        t = (e.get('fin') or time.time()) - e.get('inicio', time.time())
        if en_marcha:
            n, i = e.get('n') or 0, e.get('i') or 0
            txt = f"**{e.get('nombre_paso', '')}**" + (f" — position {e.get('pos')} ({i + 1} of {n})" if n else '')
            st.progress(min(i / n, 1.0) if n else 0.0, text=f'{txt} · {int(t // 60)} min {int(t % 60)} s')
        elif e.get('estado') == 'terminado':
            st.success(f'Finished in {int(t // 60)} min {int(t % 60)} s. See the tabs **3 · Check alignment** and **4 · Results**.')
        elif e.get('estado') == 'parado':
            st.warning('Stopped. Press Start to continue: the steps already done are skipped.')
        elif e.get('estado') == 'error':
            st.error('Stopped by an error: ' + str(e.get('mensaje', '')) + '. The messages below say where.')
        f = INTERFAZ / 'registro.txt'
        if f.exists():
            lineas = f.read_text(encoding='utf-8', errors='replace').splitlines()
            st.code('\n'.join(lineas[-30:]) or ' ', language=None)
            if len(lineas) > 30:
                st.caption(f'Last 30 of {len(lineas)} lines; all in Stacks/interfaz/registro.txt')
        if not en_marcha and st.session_state.get('estaba_en_marcha'):
            st.session_state.estaba_en_marcha = False; st.rerun()
        st.session_state.estaba_en_marcha = en_marcha
    progreso()

# ---------------------------------------------------------------------------------------------- 3. alignment
with tab_ali:
    st.header('Check the alignment')
    fr = STACKS / 'resumen_stacks.csv'
    if not fr.exists():
        st.info('Run step 1 first.')
    else:
        r = pd.read_csv(fr, dtype={'pos': str})
        cols = [c for c in ['pos', 'slices', 'calidad_min', 'residuo_max_px', 'giro_max', 'dx_min', 'dx_max', 'dy_min',
                            'dy_max', 'slices_revisar'] if c in r]
        t = r[cols].rename(columns={'calidad_min': 'min. quality', 'residuo_max_px': 'max. residual (px)',
                                    'giro_max': 'max. rotation (°)', 'slices_revisar': 'frames to check'})
        st.write('One row per position. **Quality** = how well each frame matches the reference (1 = perfect; '
                 'below 0.8, look at it). **Residual** = shift left between consecutive frames after aligning '
                 '(should be 0–1 px).')
        t = t.fillna({'frames to check': ''}) if 'frames to check' in t else t
        sty = t.style.map(lambda v: 'background-color: #f8d7da' if isinstance(v, float) and v < 0.8 else '',
                          subset=['min. quality']).format({'min. quality': '{:.3f}', 'max. residual (px)': '{:.0f}',
                                                           'max. rotation (°)': '{:.2f}'}, na_rep='')
        st.dataframe(sty, width='stretch', hide_index=True, height=min(38 * (len(t) + 1), 420))
        pos_a = st.selectbox('Position', r.pos.tolist(), key='pos_ali',
                             index=int(np.argmin(r.calidad_min.values)) if 'calidad_min' in r else 0)
        stk = next(iter(STACKS.glob(f'{pos_a}-*-BF.tif')), None)
        reg = next(iter(STACKS.glob(f'{pos_a}-*_registro.csv')), None)
        if stk and reg:
            g = pd.read_csv(reg)
            g['hora'] = g.hora.astype(str).str[11:19]
            mal = g[(g.calidad < 0.85) | (g.get('residuo_px', 0) > 2) | (g.get('encaje_anterior', 1) < 0.8)]
            if len(mal):
                st.write(f'{len(mal)} frames of position {pos_a} may be badly aligned (quality < 0.85, residual > 2 px '
                         'or poor match with the previous frame). Compare them with the reference frame: if one shows '
                         'another field of the chip or is out of focus, leave it out.')
            else:
                st.success(f'All the frames of position {pos_a} align well. You can still look at any of them below.')
            ref = int(r.loc[r.pos == pos_a, 'slice_referencia'].iloc[0]) if 'slice_referencia' in r else 1
            mt = stk.stat().st_mtime
            ver = st.multiselect('Frames to look at', g.slice.tolist(), default=mal.slice.tolist()[:6], key=f'ver_{pos_a}')
            desc = leer_descartadas(STACKS); nuevo_desc = dict(desc); duenio = duenios_fluo(inv)
            cols = st.columns(3)
            cols[0].image(miniatura(str(stk), mt, 420, ref - 1), caption=f'reference frame {ref}')
            for k, s in enumerate(ver):
                x = g[g.slice == s].iloc[0]
                col = cols[(k + 1) % 3]
                col.image(miniatura(str(stk), mt, 420, int(s) - 1),
                          caption=f'frame {s} · {x.hora} · quality {x.calidad:.2f}'
                                  + (f' · residual {x.residuo_px:.0f} px' if 'residuo_px' in x else ''))
                fuera = col.checkbox('Leave out this image', value=x.archivo_BF in desc, key=f'ali_{x.archivo_BF}')
                for a in [x.archivo_BF] + duenio.get(x.archivo_BF, []):
                    if fuera: nuevo_desc.setdefault(a, 'badly aligned (chosen in the interface)')
                    else: nuevo_desc.pop(a, None)
            if nuevo_desc != desc and st.button('Save the images to leave out', type='primary', key='guardar_ali'):
                red = guardar_descartadas(STACKS, nuevo_desc)
                st.success('Saved. ' + (f'Run step 1 again (tab 2): positions {", ".join(red)} will be redone, '
                                        'and the following steps after them.' if red else ''))
                st.rerun()
            with st.expander(f'All frames of position {pos_a}'):
                show = [c for c in ['slice', 'hora', 'archivo_BF', 'giro_grados', 'dy_px', 'dx_px', 'calidad', 'residuo_px',
                                    'encaje_anterior'] if c in g]
                st.dataframe(g[show], width='stretch', hide_index=True)

# ---------------------------------------------------------------------------------------------- 4. results
with tab_res:
    st.header('Results')
    BF = {p.name.split('-')[0]: p for p in sorted(STACKS.glob('*-BF.tif'))}
    hechos = {p: r for p, r in BF.items() if (CANALES / f"{r.stem.replace('-BF', '')}_canales_BF_linaje_ciclos.csv").exists()}
    if not hechos:
        st.info('There are no lineages yet (step 3).')
    else:
        filas = []
        for p, r in hechos.items():
            base = r.stem.replace('-BF', '')
            K = pd.read_csv(CANALES / f'{base}_canales_BF_linaje_ciclos.csv')
            ok = (K.completo.astype(bool) & K.nace_por.eq('simétrica') & K.reparto.between(0.35, 0.65)
                  & (K.Ld_px / K.L0_px).between(1.4, 2.8))
            q = K[ok]; fin = K.final.astype(str)
            fila = {'position': p, 'cells': len(K), 'divisions': int(fin.str.startswith('se divide').sum()),
                    'disappear in the channel': int(fin.str.startswith('desaparece').sum()), 'valid cycles': len(q),
                    'cycle (min)': q.duracion_min.median(), 'L birth (px)': q.L0_px.median(),
                    'L division (px)': q.Ld_px.median(), 'rate (1/min)': q.tasa_por_min.median()}
            fd = CANALES / f'{base}_canales_BF_detector_centros.csv'
            if fd.exists():
                d = pd.read_csv(fd, usecols=['foto', 'confianza'])
                fila['doubtful boxes'] = (d.confianza < 0.6).mean()
                fr_ = d.groupby('foto').confianza.apply(lambda s: (s < 0.6).mean()).rolling(3, center=True, min_periods=1).mean()
                fila['frames with poor detection'] = int((fr_ >= 0.2).sum())
            filas.append(fila)
        T = pd.DataFrame(filas).set_index('position')
        st.write('Valid cycles: born and divided in the movie, symmetric birth, 35–65 % of the mother\'s length, '
                 'Ld/L0 between 1.4 and 2.8. **Doubtful boxes**: network confidence < 0.6; **frames with poor '
                 'detection**: ≥ 20 % doubtful boxes (mean of 3 frames) — treat the numbers of those frames with caution.')
        fmt = {c: '{:.3f}' for c in ['rate (1/min)', 'doubtful boxes'] if c in T} | \
              {c: '{:.1f}' for c in ['cycle (min)', 'L birth (px)', 'L division (px)'] if c in T}
        st.dataframe(T.style.format(fmt, na_rep='–'), width='stretch', height=min(38 * (len(T) + 1), 420))
        c1, c2, c3 = st.columns(3)
        if (STACKS / 'linaje_ciclos_todas.csv').exists() and c1.button('Open the table of all cells (CSV)'):
            abrir(STACKS / 'linaje_ciclos_todas.csv')
        if c2.button('Open the Excel folder', disabled=not (STACKS / 'Excel').exists()):
            abrir(STACKS / 'Excel')
        if c3.button('Open the folder of stacks with boxes', disabled=not LINAJE.exists()):
            abrir(LINAJE)
        crec = {p: CANALES / f"{r.stem.replace('-BF', '')}_canales_BF_crecimiento_posicion.csv" for p, r in hechos.items()}
        crec = {p: pd.read_csv(f) for p, f in crec.items() if f.exists()}
        if crec:
            st.subheader('Growth rate of each position')
            df = pd.concat([g.assign(position=p) for p, g in crec.items()])
            st.line_chart(df, x='tiempo_min', y='tasa_esperada_por_min', color='position',
                          x_label='time (min)', y_label='growth rate (1/min)', height=300)

        st.subheader('One position')
        pos_r = st.selectbox('Position', list(hechos), key='pos_res')
        r = hechos[pos_r]; base = r.stem.replace('-BF', '')
        lt = LINAJE / f'{r.stem}_linaje.tif'
        if lt.exists():
            n = n_paginas(lt)
            fr_sel = st.slider('Frame', 1, n, max(1, n // 2), key=f'foto_{pos_r}')
            st.image(miniatura(str(lt), lt.stat().st_mtime, 1200, fr_sel - 1, rgb=True),
                     caption=f'{lt.name} — frame {fr_sel} of {n}. Each cell is a coloured [ ] with its number; dotted: '
                             'the network did not see it and the tracking filled it in; dashed: it reaches the exit.')
            c1, c2 = st.columns(2)
            if c1.button('Open this stack (Fiji, with Bio-Formats)', key='abrir_lin'):
                abrir(lt)
            ce = CANALES / f'{base}_canales_BF.tif'
            if c2.button('Open the straightened channels (to correct boxes with the Fiji macro)', disabled=not ce.exists()):
                abrir(ce)
        png = STACKS / 'Kimogramas' / f'{base}_kimogramas.png'
        if png.exists():
            with st.expander('Kymographs (check that every channel is there, with its closed end and exit)'):
                st.image(str(png), width='stretch')

# ---------------------------------------------------------------------------------------------- help
with tab_ayuda:
    st.header('How to use it')
    st.markdown('''
1. **Choose the folder** with the images of one experiment (panel on the left). The images must be named
   `<position>_<yymmdd>_<hhmmss>_anything.tif`. Write the names of the fluorescences if there are any.
2. **1 · Images**: check the number of images per position. If two fluorescences are equally bright (e.g. GFP and
   PI), assign them by the time of each round. Leave out repeated or wrong images (another field of the chip, out of focus).
3. **2 · Run**: choose the positions and steps and press **Start**. You can close the browser tab: it keeps running,
   and when you open the interface again you see the progress. Steps already done are skipped, so after stopping
   it just press Start again.
4. **3 · Check alignment**: positions with low quality or large residuals; look at their frames and leave out the bad ones,
   then run step 1 again (only those positions are redone).
5. **4 · Results**: summary per position, growth rate, and the stacks with one box per cell to check the lineage.

**Correcting boxes by hand** is done in Fiji with the macro `Cajas_Canales.ijm` (see `docs/training.md`): open the
straightened channels of a position, correct the boxes, and run step 3 again: the positions you corrected are redone
with your boxes.

**Where everything goes**: `Stacks/` next to the images folder (see `docs/outputs.md` for every file and column).
The images themselves are never modified.

**If something fails**: the messages of the last run are in `Stacks/interfaz/registro.txt`.
''')
    st.caption(f'Programs: {SCRIPTS}')
