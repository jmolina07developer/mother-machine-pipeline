# Using the pipeline

The programs are in `scripts/` and must be run from that folder (they import each other). Every step can
be run from the notebook `pipeline_completo.ipynb` or from the terminal; the result is the same.

## Input

**One folder per experiment** containing the raw images of all positions, for example:

```
MyExperiment/
  Images/
    01_260925_103512_Regrowth_BF_00300_.tif
    01_260925_103515_Regrowth_BF_00300_.tif
    02_260925_103530_Regrowth_BF_00300_.tif
    …
```

Results are written to the sibling folder `MyExperiment/Stacks/`. The raw images are never modified.

**File names** must be `<position>_<yymmdd>_<hhmmss>_<anything>.tif` (the pattern `PATRON` in
`generar_stacks.py`; change it there if your microscope names files differently). The position and the
real time of each image are read from the name; real times are used for every rate.

**Images.**

- One 2D TIFF per image (8- or 16-bit), all positions and times in the same folder.
- Channels run vertically. The numbers engraved on the chip must be visible in the upper part of the image
  (rows 0–450, `ZONA_NUMEROS` in `alinear_stack.py`): they are used to align the images over time.
- Bright-field and fluorescence images may have the same kind of name. They are told apart by their
  brightness (fluorescence images are much darker) and paired in time: each fluorescence image belongs to
  the bright-field image taken less than 60 s before or after it (`MAX_SEGUNDOS`).
- With several fluorescence channels (e.g. GFP and PI), give them from the brightest to the dimmest
  (`--fluo GFP PI`, or `FLUO = ['GFP', 'PI']` in the notebook). They are separated by brightness.
- A fluorescence channel does not need to be imaged at every time point: missing slices are left black
  in its stack and are not measured.
- Positions with fewer than 10 images (`POS_MIN_IMAGENES`) are ignored (tests, stray images).

## The notebook

Open `scripts/pipeline_completo.ipynb` from the `scripts` folder (`cd scripts; jupyter lab`) and edit the
*Parameters* cell:

| Parameter | Meaning |
|---|---|
| `CARPETA_IMAGENES` | the `Images` folder of the experiment |
| `POSICIONES` | `None` for all, or a list such as `['01', '02']` |
| `SOLO_BF` | `None` = detect automatically whether there are fluorescence images; `True` = bright field only |
| `FLUO` | fluorescence name(s): `'PI'`, `'GFP'` or a list from the brightest to the dimmest |
| `REHACER` | `True` to redo every step even if its results exist |
| `PASOS` | switch steps on or off |

Then *Run All*. Each step skips the positions already done and redoes a position when its input changed
(a newer stack, your corrected boxes, newer programs or a retrained network).

## The steps from the terminal

### 1. Aligned stacks: `generar_stacks.py`

```bash
python generar_stacks.py /path/to/MyExperiment/Images [--fluo GFP PI] [--posiciones 01 02] [--solo-bf]
```

For every position:

- it pairs bright-field and fluorescence images;
- it aligns all of them on the chip numbers (integer shifts, so nothing is blurred), and
  equalises the bright-field exposure if it changes during the movie;
- it writes `Stacks/<pos>-<ddmmyyyy>-BF.tif` (8-bit) and `-<FLUO>.tif` (16-bit), with a table
  `<pos>-<ddmmyyyy>_registro.csv`.

It also checks that each image belongs to its position (if a round of positions was saved with shifted
numbers, the images are moved to the right position; listed in `Stacks/areas_reasignadas.csv`). It
discards duplicated bright-field images taken seconds apart, keeping the one that has the fluorescence.

**Large stage jumps.** The normal search looks for shifts of up to `--margen` px. Slices that still match
poorly (quality < 0.93) are searched again over all shifts of up to 500 px, so that a stage that moved by a
few hundred pixels is aligned correctly instead of locking onto the neighbouring block of chip numbers
(they repeat every ~200 px). After such a jump part of the aligned slice has no image (it is filled with
grey); step 2 lists the channels and frames concerned and the tracking skips them.

**Discarding images by hand.** To leave out images (e.g. a frame of another field of the chip saved under
a position), list them in `Stacks/imagenes_descartadas.csv`:

```
archivo,motivo
08_261103_160311_Regrowth_BF_00300_.tif,another field of the chip
```

and run step 1 again for those positions.

**Check:** the printed line of each position. `calidad` (alignment quality, correlation with the
reference, 1 = perfect) below 0.8, or `DOUBTFUL ALIGNMENT in slices …`, means you should look at the
stack in Fiji.

Options: `--margen` (maximum shift searched, px, default 80), `--sin-alinear`, `--bf-16bits`,
`--sin-corregir-areas`, `--salida` (output folder).

### 2. Channels: `canales_enderezados.py`

```bash
python canales_enderezados.py /path/to/MyExperiment/Stacks/*-BF.tif
```

It first runs `kimogramas.py` and then makes the straightened channels:

- `kimogramas.py` detects the channels and numbers them from left to right, and finds the closed end
  and the exit of each one. It writes the kymographs to `Stacks/Kimogramas/`; check
  `<pos>_kimogramas.png`: all channels should be there, with the closed end and the exit in place.
- `canales_enderezados.py` then writes `Stacks/Canales/<pos>_canales_BF.tif`. It has one slice per
  channel, with all its frames side by side, straightened, with the closed end at the top. This is the
  image the Fiji macro works on.
- It also writes `<pos>_canales_BF_geometria.csv`. Channels cut by the edge of the image are marked
  `al_borde` and not analysed.
- `<pos>_canales_BF_sin_datos.csv` lists the channels and frames that fall outside the real image after a
  large stage jump (grey in the straightened image). They are treated as bad frames of that channel: the
  tracking skips them and they are not used for training.

### 3. Cells and lineage: `detectar_celulas.py predecir`

```bash
python detectar_celulas.py predecir /path/to/MyExperiment/Stacks/Canales/*_canales_BF.tif
```

**Detection.** The centre-based neural network (`modelos/detector_centros.onnx`, run with onnxruntime)
proposes a box for every cell in every frame. It looks at each frame together with the two frames
before and after it.

**Tracking.** For each channel, the boxes are tracked over the **whole movie at once** by integer
programming (`linaje.py`):

- cells keep their order in the channel (they cannot overtake each other);
- each cell can continue, divide (symmetrically, or polar/fragmentation for filaments), leave through
  the exit or disappear inside the channel (death, lysis or loss);
- the network may miss a cell in a frame (its box is filled in, `interpolada`), or see a box that is not
  a cell;
- cells below an empty gap are followed too.

The result is the lineage of every cell: `_linaje.csv` (cell × frame) and `_linaje_ciclos.csv` (one row
per cell). It also writes a list of frames worth reviewing in Fiji (`_revisar.txt`) and the boxes
proposed to the macro (`_detector.txt`). Takes ~1–2 min per position.

**Your corrections.** In the frames where you corrected or drew boxes with the Fiji macro
(`<pos>_canales_BF_cajas.zip`), your boxes are used instead of the network's (column `manual` in
`_linaje.csv`).

**Bad frames.** Frames marked as bad with the macro (key `6`) are skipped. If the end of a movie is
unusable (e.g. the medium ran out), create `Stacks/Canales/fotos_validas.csv`:

```
imagen,ultima_foto,motivo
04-26092025,70,medium ran out
```

**Other subcommands** (mostly for annotation and training, see [training.md](training.md)):

| Subcommand | What it does |
|---|---|
| `proponer` | write only the boxes proposed to the Fiji macro (key `8`) and their doubts, without redoing the lineage |
| `revision` | rank the frames where the tracking is least certain (`-n` per image) |
| `entrenar`, `validar`, `probar` | train / cross-validate / test the tree-based detector with your boxes |
| `figura` | figure of one channel (`--canal 3 --fotos 60-100`) |

Settings at the top of `detectar_celulas.py`:

- `SEGUIMIENTO`: `'global'` (default) or `'foto_a_foto'` (faster, slightly worse);
- `PROCESOS`: channels tracked in parallel (default 4);
- `USAR_CAJAS_MANUALES`: whether to use your corrected boxes (default on).

Without onnxruntime the older tree-based detector (`modelos/detector_canales.joblib`) is used instead;
it is slower and less accurate.

### 4. Stacks with boxes: `stack_linaje.py`

```bash
python stack_linaje.py /path/to/MyExperiment/Stacks/04-26092025-BF.tif [--fotos 1-40]
```

`Stacks/Linaje/<pos>-<ddmmyyyy>-BF_linaje.tif` is the colour movie with one box and one number per
cell. The number is `celula` in the tables, unique within each channel; daughters get new numbers and
colours. In the boxes:

- dashed boxes are cells reaching the exit;
- dotted boxes are frames in which the network missed the cell and the tracking filled it in.

Each frame shows its number and time. The file is losslessly compressed: open it in Fiji with
*File › Import › Bio-Formats*.

### 5. Mothers' sheet: `excel_madres.py`

```bash
python excel_madres.py /path/to/MyExperiment/Stacks/*-BF.tif
```

`Stacks/Excel/<pos>-<ddmmyyyy>_longitud_madres.xlsx` has one sheet per position:

- row 1: the time of each frame; row 2: the frame number;
- then two rows per channel (column A = channel number, as drawn on the Linaje stacks):
  - upper row: the length (px) of the mother cell (the cell at the closed end) in each frame;
  - lower row: the length of the daughter, only in the first frame where a division of the mother is
    seen, so that mother + daughter = the mother before dividing.

Colours of the empty cells:

- red: the cell goes past the exit (measure it by hand);
- yellow: the cell is there but its length is not reliable;
- grey: bad frame.

Green cells hold lengths from frames you corrected by hand. The sheet is overwritten when the lineage
changes: save your manual measurements in a copy.

### 6. Fluorescence per cell: `fluorescencia_linaje.py`

```bash
python fluorescencia_linaje.py /path/to/MyExperiment/Stacks/*-BF.tif --fluo GFP
```

For every cell of the lineage in every frame with a fluorescence image, it measures:

- the mean intensity of the central columns of the channel between the cell's poles (2 px removed at
  each pole);
- the background of the channel walls in the same rows and frame, subtracted;
- the background of the medium (empty stretches of the channels), subtracted.

The result, `fluo_neta`, is proportional to the concentration; `fluo_total` = `fluo_neta` × length is
proportional to the amount. Written to `Stacks/Canales/<pos>_canales_BF_linaje_<FLUO>.csv`.

The medium background is estimated per position from the stretches of channel without a box. If the
detector misses many cells (e.g. dying cells), check the column `fondo_medio`: an estimate far above the
other positions means that unboxed cells were taken as background.

## Checking the results

- **Alignment**: `calidad` < 0.8 or "doubtful alignment" → look at the stack.
- **Channels**: `Stacks/Kimogramas/<pos>_kimogramas.png`.
- **Boxes**: open a few `Stacks/Linaje/*_linaje.tif` and go through the frames. If the boxes are wrong in
  many frames, the detector needs examples of your images ([training.md](training.md)).
- **Quality of the detection over time**: `_detector_centros.csv` has the confidence of every box (column
  `confianza`; `dudosa` = confidence < 0.6). A sustained increase of the fraction of doubtful boxes
  usually means that the cells look different (e.g. dying, granular cells) and that the numbers of that
  period should be treated with caution.
- **Review list**: in Fiji, key `9` goes through the frames listed in `_revisar.txt`.

## Units

Lengths in pixels (rows of the straightened channel), times in minutes or clock time, rates in 1/min
(exponential: Δln L / Δt), fluorescence in camera counts above background.
