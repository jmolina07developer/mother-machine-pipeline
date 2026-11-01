"""Check that everything the pipeline needs is installed and works on this computer.

Usage (from the scripts folder, with the environment activated):
    python check_install.py

It checks the Python version, the required packages and their versions, that the trained models load,
that the neural network runs (onnxruntime), that the integer-programming solver used by the tracking is
available, that every program of the pipeline can be imported, and (optionally) the packages needed to
retrain the network. It does not modify anything.
"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# versions with which the pipeline has been tested (requirements.txt)
TESTED = {'numpy': '2.2.6', 'pandas': '2.3.3', 'scipy': '1.15.3', 'skimage': '0.25.2', 'sklearn': '1.7.2',
          'tifffile': '2025.5.10', 'roifile': '2025.12.12', 'onnxruntime': '1.23.2', 'PIL': '12.3.0',
          'joblib': '1.6.0', 'openpyxl': '3.1.5', 'matplotlib': '3.10.9'}
PIP_NAME = {'skimage': 'scikit-image', 'sklearn': 'scikit-learn', 'PIL': 'pillow'}
problems, warnings_ = [], []


def ok(msg): print(f'  [ok]   {msg}')
def warn(msg): print(f'  [warn] {msg}'); warnings_.append(msg)
def fail(msg): print(f'  [FAIL] {msg}'); problems.append(msg)


print('1. Python')
v = sys.version_info
if v < (3, 10):
    fail(f'Python {v.major}.{v.minor} is too old: use Python 3.10, 3.11 or 3.12')
elif v >= (3, 13):
    warn(f'Python {v.major}.{v.minor}: not tested yet (tested with 3.10 and 3.12)')
else:
    ok(f'Python {v.major}.{v.minor}.{v.micro}')

print('2. Packages')
for mod, tested in TESTED.items():
    try:
        m = importlib.import_module(mod)
        have = getattr(m, '__version__', '?')
        if mod == 'sklearn' and have != tested:
            fail(f'scikit-learn {have}: the tree-based models were saved with {tested} and may not load '
                 f'(pip install scikit-learn=={tested})')
        elif have.split('.')[0] != tested.split('.')[0]:
            warn(f'{PIP_NAME.get(mod, mod)} {have} (tested with {tested})')
        else:
            ok(f'{PIP_NAME.get(mod, mod)} {have}')
    except ImportError:
        fail(f'{PIP_NAME.get(mod, mod)} is not installed (pip install {PIP_NAME.get(mod, mod)}=={tested})')
if problems:
    print('\nInstall the missing packages first:  pip install -r requirements.txt')
    sys.exit(1)

print('3. Trained models (scripts/modelos)')
import joblib
import numpy as np
for name, needed in (('detector_centros.onnx', True), ('detector_canales.joblib', True), ('bordes_kimo.joblib', True),
                     ('detector_centros.pt', False)):
    p = HERE / 'modelos' / name
    if not p.exists():
        (fail if needed else warn)(f'{name} not found in {p.parent}')
        continue
    if name.endswith('.joblib'):
        try:
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always'); joblib.load(p)
            other = sorted({str(x.message).split('from version ')[1].split(' ')[0] for x in w
                            if 'InconsistentVersionWarning' in type(x.message).__name__ and 'from version ' in str(x.message)})
            ok(f'{name} loads' + (f' (saved with scikit-learn {", ".join(other)}; works with {TESTED["sklearn"]})' if other else ''))
        except Exception as e:  # noqa: BLE001
            fail(f'{name} does not load ({type(e).__name__}: {e}); check the scikit-learn version')
    else:
        ok(f'{name} present ({p.stat().st_size / 1e6:.1f} MB)')

print('4. Neural network (onnxruntime)')
try:
    import centros_comun as CC
    det = CC.DetectorCentros()
    tiles = np.random.default_rng(0).integers(0, 255, (6, CC.LARGO, CC.ANCHO), dtype=np.uint8)
    C, D, P = det.mapas(tiles, 220, 10, 230)
    assert C.shape == (6, 220) and D.shape == (6, 2, 220) and P.shape == (6, 220, 4)
    ok('the centre-based detector runs on a test image')
except Exception as e:  # noqa: BLE001
    fail(f'the centre-based detector does not run ({type(e).__name__}: {e})')

print('5. Integer-programming solver for the tracking (scipy.optimize.milp, HiGHS)')
try:
    from scipy.optimize import milp, LinearConstraint
    r = milp(c=[-1, -1], constraints=LinearConstraint([[1, 1]], 0, 1), integrality=[1, 1], bounds=(0, 1))
    assert r.success
    ok('milp works')
except Exception as e:  # noqa: BLE001
    fail(f'scipy.optimize.milp does not work ({type(e).__name__}: {e}); scipy >= 1.9 is needed')

print('6. Programs of the pipeline')
import matplotlib
matplotlib.use('Agg')
for mod in ('alinear_stack', 'generar_stacks', 'kimogramas', 'canales_enderezados', 'centros_comun', 'segmentar_kimogramas',
            'linaje', 'leer_cajas', 'leer_roiset', 'kimo_trazos', 'detectar_celulas', 'stack_linaje', 'excel_madres',
            'fluorescencia_linaje', 'marcar_canales_vacios'):
    try:
        importlib.import_module(mod); ok(mod)
    except Exception as e:  # noqa: BLE001
        fail(f'{mod} cannot be imported ({type(e).__name__}: {e})')

print('7. Jupyter (to use pipeline_completo.ipynb)')
try:
    import IPython  # noqa: F401
    import ipykernel  # noqa: F401
    ok('Jupyter kernel available')
except ImportError:
    warn('Jupyter is not installed: the programs work from the terminal, but to use the notebook run  pip install jupyterlab')

print('8. Optional: retraining the network (PyTorch + onnx)')
try:
    import torch  # noqa: F401
    import onnx  # noqa: F401
    import detector_centros  # noqa: F401
    ok(f'torch {torch.__version__} and onnx {onnx.__version__}: retraining available')
except ImportError:
    print('  [--]   not installed (only needed to retrain the network: pip install -r requirements-training.txt)')

print()
if problems:
    print(f'{len(problems)} problem(s) found:')
    for p in problems: print('  -', p)
    sys.exit(1)
print('Everything needed is installed.' + (f' ({len(warnings_)} warning(s) above.)' if warnings_ else ''))
