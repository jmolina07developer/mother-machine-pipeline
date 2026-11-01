# Installation

Everything runs on a normal laptop or desktop (macOS, Windows or Linux), without a GPU. You need:

- **Python 3.10, 3.11 or 3.12** with the packages in `requirements.txt` (the pipeline itself);
- **Jupyter** (installed with the packages) to use the notebook;
- **Fiji** to look at the results and to review or annotate boxes (optional but recommended);
- **PyTorch** only if you want to retrain the neural network (optional).

The steps below use **Miniforge** (a small, free conda distribution), which installs its own Python and
keeps the pipeline in a separate environment, so it does not interfere with anything else on the
computer. A plain-Python alternative is given in step 3b.

Tested with Python 3.10, 3.11 and 3.12 and the package versions in `requirements.txt`.

---

## 1. Install Miniforge (once per computer)

**macOS and Linux.** Open the Terminal and run:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Accept the licence, keep the default location and answer **yes** when asked whether to initialise conda.
Close the Terminal and open a new one: the prompt now starts with `(base)`.

**Windows.** Download `Miniforge3-Windows-x86_64.exe` from
<https://github.com/conda-forge/miniforge/releases/latest> and install it with the default options
("Just me"). Then open **Miniforge Prompt** from the Start menu and use it for all the commands below.

> If you already have Anaconda or Miniconda, you can use it instead: skip this step.

## 2. Download the pipeline

With git (`git --version` tells you whether it is installed; on macOS it is offered automatically the first
time you type `git`; on Windows install it from <https://git-scm.com>):

```bash
cd ~/Documents            # or wherever you want to keep it
git clone https://github.com/jmolina07developer/mother-machine-pipeline.git
cd mother-machine-pipeline
```

Without git: on the GitHub page press **Code › Download ZIP**, unzip it and `cd` into the folder.

## 3. Create the environment

```bash
conda env create -f environment.yml
conda activate mothermachine
```

This installs Python 3.12 and all the packages (a few minutes). **Every time** you open a new terminal to
use the pipeline, run `conda activate mothermachine` first.

### 3b. Alternative without conda

Install Python 3.12 from <https://www.python.org/downloads/> (on Windows tick *Add python.exe to PATH*),
then, inside the `mother-machine-pipeline` folder:

```bash
python3 -m venv .venv                 # Windows: py -3.12 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Activate it (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows) every time you use the
pipeline.

## 4. Check the installation

```bash
cd scripts
python check_install.py
```

It checks the packages and their versions, loads the trained models, runs the neural network on a test
image and imports every program. It must end with:

```
Everything needed is installed.
```

Each `[FAIL]` line says what is wrong and how to fix it. `[warn]` lines (e.g. a slightly different
package version) can usually be ignored.

## 5. Open the notebook

```bash
cd scripts                # the notebook must be opened from this folder
jupyter lab
```

The browser opens Jupyter; double-click `pipeline_completo.ipynb`, edit the *Parameters* cell and use
*Run › Run All Cells*. See [docs/usage.md](docs/usage.md).

Paths in the notebook: on Windows write them with forward slashes or as raw strings, e.g.
`Path('C:/Data/MyExperiment/Images')` or `Path(r'C:\Data\MyExperiment\Images')`.

## 6. Fiji and the review macro (recommended)

1. Download Fiji from <https://fiji.sc> (*Download*), unzip it and open it. On macOS, if it says that the
   application cannot be opened, right-click on `Fiji.app` › *Open*.
2. Install the macro `fiji/Cajas_Canales.ijm`, in one of two ways:
   - **for this session**: *Plugins › Macros › Install…* and choose the file (repeat each time Fiji is
     opened);
   - **permanently**: copy `Cajas_Canales.ijm` into `Fiji.app/macros/toolsets/`, restart Fiji and choose
     *Cajas_Canales* in the `>>` menu at the right end of the toolbar.
3. Open a straightened-channel image (`Stacks/Canales/<pos>_canales_BF.tif`) and press `0` for the
   list of keyboard shortcuts. How to use it: [docs/training.md](docs/training.md).

The stacks with boxes (`Stacks/Linaje/*_linaje.tif`) are compressed: open them with
*File › Import › Bio-Formats* (or drag them onto Fiji).

## 7. Optional: retraining the neural network

Only needed if the detector does not work well on your images (see [docs/training.md](docs/training.md)):

```bash
conda activate mothermachine
pip install -r requirements-training.txt
python scripts/check_install.py        # section 8 now says "retraining available"
```

The CPU version of PyTorch is enough (training takes ~10–30 min on a laptop). On Windows, if `pip`
cannot find a PyTorch version for your Python, follow the command given at <https://pytorch.org/get-started/locally/>.

## Troubleshooting

- **A pinned version cannot be installed on my system** (very old or very new computer). Install the
  packages without fixed versions, except scikit-learn, and run `check_install.py`:
  ```bash
  pip install numpy pandas scipy scikit-image "scikit-learn==1.7.2" tifffile roifile onnxruntime pillow joblib openpyxl matplotlib jupyterlab
  ```
- **scikit-learn warning or the tree-based model does not load.** The `.joblib` models need scikit-learn
  1.7.2: `pip install scikit-learn==1.7.2`, then restart the Jupyter kernel. (The neural network, used by
  default, does not depend on scikit-learn.)
- **`onnxruntime` is missing.** The pipeline then falls back to the older tree-based detector, which is
  slower and less accurate. Install it with `pip install onnxruntime`.
- **The notebook says a package is missing although it is installed.** Jupyter is using another Python.
  Start Jupyter from the activated environment (`conda activate mothermachine`, then `jupyter lab`), or
  register the environment as a kernel and choose it in *Kernel › Change Kernel*:
  ```bash
  python -m ipykernel install --user --name mothermachine
  ```
- **"Open this notebook from the scripts folder".** Jupyter was started elsewhere: `cd scripts` before
  `jupyter lab`.
- **After updating the programs the notebook still behaves as before.** Restart the kernel
  (*Kernel › Restart*).
- **Time and memory.** Step 3 takes ~1–2 min per position (~25 channels × 70 frames) on a laptop; the
  whole pipeline for 30 positions takes about an hour. 8 GB of RAM are enough.
