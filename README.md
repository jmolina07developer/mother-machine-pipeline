# Mother machine pipeline

From the raw images of a *mother machine* (microfluidic single-cell device) to the lineage of every
bacterial cell: stacks aligned over time, straightened channels, cell detection with a neural network,
tracking of every cell through the whole movie (divisions, cells leaving the channel, deaths), growth
rates, a review sheet of the mother cells and the fluorescence of every cell.

It was developed for *E. coli* in bright field (with optional fluorescence: PI, GFP…) and has been used
on movies of 3–10 h with 15–50 positions per experiment.

## What it does

| Step | Program | Result |
|---|---|---|
| 1. Aligned stacks | `generar_stacks.py` | one stack per position (bright field + fluorescence), aligned on the numbers of the chip; checks that every image belongs to its position |
| 2. Channels | `kimogramas.py`, `canales_enderezados.py` | channels detected and numbered, kymographs, one straightened image per channel and frame |
| 3. Cells and lineage | `detectar_celulas.py predecir` (+ `linaje.py`) | boxes of all cells in every frame (centre-based neural network), tracked over the whole movie at once by integer programming; one row per cell with its mother, birth, division or fate, lengths and growth rate |
| 4. Stacks with boxes | `stack_linaje.py` | the movie with a coloured, numbered box on every cell, to check the result in Fiji |
| 5. Mothers' sheet | `excel_madres.py` | an Excel sheet with the length of the mother cell of every channel in every frame, colour-coded for manual review |
| 6. Fluorescence | `fluorescencia_linaje.py` | background-corrected fluorescence of every cell in every frame |
| Review and annotation | `fiji/Cajas_Canales.ijm` | Fiji macro to check and correct boxes, mark bad frames and annotate new training data |

All the steps can be run from one Jupyter notebook (`scripts/pipeline_completo.ipynb`) or one by one
from the terminal. Each step skips what is already done, so an interrupted run can simply be restarted.

## Installation

Short version (details, for macOS, Windows and Linux, in **[INSTALL.md](INSTALL.md)**):

```bash
git clone https://github.com/jmolina07developer/mother-machine-pipeline.git
cd mother-machine-pipeline
conda env create -f environment.yml        # or: python -m venv .venv  +  pip install -r requirements.txt
conda activate mothermachine
cd scripts
python check_install.py                    # must end with "Everything needed is installed."
```

To review boxes and annotate, install [Fiji](https://fiji.sc) and the macro `fiji/Cajas_Canales.ijm`
(see INSTALL.md).

## Quick start

1. Put the raw images of **one experiment** in a folder called `Images`, named as the microscope saves
   them: `<position>_<yymmdd>_<hhmmss>_<anything>.tif`, e.g. `04_260925_103958_Regrowth_BF_00300_.tif`.
2. Open `scripts/pipeline_completo.ipynb` in Jupyter (from the `scripts` folder), set `CARPETA_IMAGENES`
   (the `Images` folder) and `FLUO` (fluorescence channel(s), if any) in the *Parameters* cell, and run
   all the cells.
3. Results are written next to the images, in `<experiment>/Stacks/`.

Or from the terminal (same steps):

```bash
cd scripts
python generar_stacks.py /path/to/MyExperiment/Images --fluo GFP PI
python canales_enderezados.py /path/to/MyExperiment/Stacks/*-BF.tif
python detectar_celulas.py predecir /path/to/MyExperiment/Stacks/Canales/*_canales_BF.tif
python stack_linaje.py /path/to/MyExperiment/Stacks/04-26092025-BF.tif
python excel_madres.py /path/to/MyExperiment/Stacks/*-BF.tif
python fluorescencia_linaje.py /path/to/MyExperiment/Stacks/*-BF.tif --fluo GFP
```

See **[docs/usage.md](docs/usage.md)** for what each step expects and does, its options and how to
check the results, and **[docs/outputs.md](docs/outputs.md)** for every output file and column.
The algorithms behind each step (registration, the cell detector, the integer program used for tracking,
the fluorescence measurement), their validation and the references they build on are described in
**[docs/methods/methods.pdf](docs/methods/methods.pdf)** (LaTeX source and figures in the same folder).

## Before trusting the numbers

The detector was trained on bright-field images of *E. coli* from one microscope and one chip design. On different images (other magnification, chip, strain or focus) it may fail: open a few
`Stacks/Linaje/*_linaje.tif` in Fiji and check the boxes. If they are wrong, correct some frames with the
Fiji macro and retrain the detector (**[docs/training.md](docs/training.md)**).

## Language of the files

The documentation, comments and messages are in English. The names of the programs, of the output files
and of the table columns are in Spanish (e.g. `canal` = channel, `foto` = frame, `celula` = cell), as are
some values inside the tables (e.g. the fate of a cell, `se divide` = divides). They are kept as they are
so that existing analyses keep working; [docs/outputs.md](docs/outputs.md) translates all of them.

## Repository layout

```
scripts/                 the programs (run them from this folder)
  pipeline_completo.ipynb  the whole pipeline in a notebook
  check_install.py         checks the installation
  modelos/                 trained models (neural network and tree-based detectors)
fiji/Cajas_Canales.ijm   Fiji macro to review and annotate boxes
docs/                    usage, outputs and training guides
  methods/                 methods document (PDF, LaTeX source and figures)
environment.yml          conda environment
requirements.txt         Python packages (requirements-training.txt adds PyTorch, to retrain)
```

## License

GNU General Public License v3.0 (see [LICENSE](LICENSE)): you may use, modify and redistribute the code;
modified versions that you distribute must also be released under the GPL-3.0.

## Authorship and AI assistance

The code, the trained models and this documentation were written by **Claude** (Anthropic's AI model), working
with **Javier Molina**, who:

- designed the experiments and provided the microscope images;
- drew the hand annotations used to train and validate the cell detector;
- decided what every step had to do;
- checked the results against his own measurements.

All the programs were tested on real experiments before release, but, as with any analysis software, check
the outputs on your own data (see *Before trusting the numbers* above).

If you use this pipeline in your work, please cite this repository.
