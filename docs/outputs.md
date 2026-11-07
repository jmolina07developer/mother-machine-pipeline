# Output files and columns

All results go to `<experiment>/Stacks/`. File names and column names are in Spanish; this page gives
their meaning. `<pos>` is the position (`04`), `<date>` is `ddmmyyyy`, `<base>` is `<pos>-<date>`.

**Coordinates.** Rows (`fila_*`) are measured along the straightened channel. Row 0 is 10 px above the
closed end (`fila_fondo` ≈ 10) and the exit is at `fila_salida`. Lengths are in pixels, times in
minutes from the first frame (`tiempo_min`, `*_min`) or as clock time (`hora`).

**Frames.** `foto` and `slice` are frame numbers, starting at 1; the same number is the same time point
in every file of a position. **Channels:** `canal` is numbered from left to right, as drawn on the
stacks and kymographs.

## Common words

| Spanish | English | | Spanish | English |
|---|---|---|---|---|
| canal | channel | | madre | mother |
| foto, slice | frame | | hija | daughter |
| celula | cell | | fondo | closed end (or background, in fluorescence) |
| caja | box | | salida | exit of the channel |
| longitud | length | | tasa | rate |
| fila | row | | nace / divide | is born / divides |
| sup / inf | top / bottom | | hora | clock time |
| dudosa | doubtful | | confianza | confidence |

## `Stacks/` (step 1)

| File | Content |
|---|---|
| `<base>-BF.tif` | aligned bright-field stack, 8-bit; slice labels = original file names |
| `<base>-<FLUO>.tif` | aligned fluorescence stack, 16-bit; slices without an image are black and labelled `SIN_<FLUO>` |
| `<base>_registro.csv` | one row per slice (below) |
| `resumen_stacks.csv` | one row per position: slices, unpaired / discarded images, alignment range and quality |
| `areas_comprobacion.csv`, `areas_reasignadas.csv` | match of every image with the chip numbers of its position and the images moved to another position |
| `imagenes_descartadas.csv` | optional, written by you: images to leave out (`archivo, motivo`) |

`_registro.csv`:

| Column | Meaning |
|---|---|
| `slice`, `hora` | frame number and clock time of the bright-field image |
| `archivo_BF`, `archivo_<FLUO>` | source files (`archivo_PI` when there is a single fluorescence channel) |
| `dt_BF_<FLUO>_s` | seconds between the bright-field image and its fluorescence image |
| `q_area` | correlation of the image with the chip numbers of its position |
| `giro_grados` | rotation of the slice relative to the median orientation of the stack (corrected if ≥ 0.1°) |
| `dy_px`, `dx_px` | shift applied to align the slice |
| `calidad` | alignment quality (correlation with the reference, 1 = perfect; < 0.8: check) |
| `comparado_con`, `salto_px` | reference slice and size of the shift relative to the previous slice |
| `residuo_px`, `encaje_anterior` | check after aligning: residual shift relative to the previous slice (should be 0–1 px) and its correlation |

## `Stacks/Kimogramas/` (step 2)

| File | Content |
|---|---|
| `<base>_kimogramas.npz` | raw kymographs (channel × position along the channel × time), channel geometry and times (load with `numpy.load(..., allow_pickle=True)`) |
| `<base>_kimo_BF.tif` | one kymograph per channel, for Fiji |
| `<base>_canales.csv` | channel geometry (same columns as `_geometria.csv` below) |
| `<base>_kimogramas.png` | all kymographs with the closed end and exit, for a quick check |

## `Stacks/Canales/` (steps 2, 3 and 6)

| File | Content |
|---|---|
| `<base>_canales_BF.tif` | straightened channels: one slice per channel, its frames side by side (open it with the Fiji macro) |
| `<base>_canales_BF_geometria.csv` | geometry of each channel (below) |
| `<base>_canales_BF_macro.txt` | crop sizes read by the Fiji macro |
| `<base>_canales_BF_linaje.csv` | **every cell in every frame** (below) |
| `<base>_canales_BF_linaje_ciclos.csv` | **one row per cell** (below) |
| `<base>_canales_BF_crecimiento_posicion.csv` | `foto`, `tiempo_min`, `tasa_esperada_por_min`: growth rate of the whole position frame by frame (used by the tracking) |
| `<base>_canales_BF_linaje_<FLUO>.csv` | fluorescence of every cell in every frame (step 6, below) |
| `<base>_canales_BF_detector.txt` | boxes proposed to the Fiji macro: `channel frame top bottom origin doubtful` |
| `<base>_canales_BF_revisar.txt` | frames worth reviewing: `channel frame reason` (reasons in Spanish, e.g. `caja dudosa` = doubtful box, `madre incierta` = uncertain mother, `foto mala (saltada)` = bad frame, skipped) |
| `<base>_canales_BF_detector_centros.csv` | raw boxes of the network: `canal, foto, orden, fila_sup, fila_inf, longitud_px, confianza, dudosa, sale_del_canal` |
| `_detector_cajas.csv`, `_detector_madre_hija.csv`, `_detector_ciclos.csv`, `_detector_divisiones.csv` | intermediate results of the mother/daughter tracking (diagnostics) |
| `<base>_canales_BF_cajas.zip` | **your** boxes and marks, written by the Fiji macro (ROI names `C05_f023_c02` = box 2 of channel 5 in frame 23; `_VACIA` = empty; `_MALA` = bad frame) |
| `<base>_canales_BF_borradas.txt` | proposals you deleted in the macro (there is no cell there; used for training) |
| `<base>_canales_BF_sin_datos.csv` | `canal, foto`: channel-frames outside the real image after a large stage jump (treated as bad frames) |
| `fotos_validas.csv` | optional, written by you: last usable frame of each image (`imagen, ultima_foto, motivo`) |
| `conjunto_prueba.csv` | optional: channels kept out of training as a test set (`imagen, canal`) |

`_geometria.csv`:

| Column | Meaning |
|---|---|
| `x_sup, y_sup, x_inf, y_inf` | ends of the channel axis in the stack (px) |
| `al_borde` | the channel is cut by the edge of the image (not analysed) |
| `corregido_sup`, `corregido_inf` | the closed end / exit deviated from the straight line through all the channels and was put back on it |
| `longitud_px` | length of the channel, closed end to exit |
| `fila_fondo`, `fila_salida` | rows of the closed end and of the exit in the straightened channel |
| `canal_usuario` | channel number in older manual measurements (if any) |
| `margen_sup, ancho, sep, largo, fotos, stack` | crop geometry of the straightened image and number of frames |

`_linaje.csv` (one row per cell and frame):

| Column | Meaning |
|---|---|
| `canal, foto, tiempo_min` | channel, frame, minutes since the first frame |
| `celula` | cell number, unique within the channel (as drawn on the Linaje stack) |
| `orden` | position in the channel in that frame (1 = at the closed end, the mother) |
| `fila_sup`, `fila_inf` | top and bottom pole |
| `longitud_px` | length; empty if the cell reaches the exit, the box was filled in, or the length is not reliable |
| `sale_del_canal` | the bottom pole is beyond the exit (the length is cut) |
| `fila_inf_visible` | bottom pole, clipped at the exit |
| `confianza` | confidence of the network for the box |
| `visible` | the box looks like a cell (False for empty channels and false detections: ignore those rows) |
| `interpolada` | the network did not see the cell in this frame and the tracking filled in its box (no length) |
| `manual` | the box is one of yours (corrected or drawn in the Fiji macro) |
| `canal_vacio` | an empty channel that the network took for one long cell (made not visible) |
| `saltada`, `seguida` | frame skipped by the tracking / cell tracked (kept for compatibility) |

`_linaje_ciclos.csv` (one row per cell):

| Column | Meaning |
|---|---|
| `canal, celula` | channel and cell number |
| `madre` | number of its mother (empty for cells present in the first frame) |
| `generacion` | number of divisions since the first frame |
| `orden_al_nacer` | position in the channel at birth |
| `nace_foto, nace_min` | frame and minute of birth |
| `divide_foto, divide_min` | frame and minute of its division |
| `final` | fate (values below) |
| `duracion_min` | cycle duration (birth to division) |
| `L0_px`, `Ld_px` | length at birth and at division |
| `tasa_por_min` | exponential growth rate over the cycle, Δln L/Δt (1/min) |
| `reparto` | fraction of the mother's length it received |
| `nace_por` | kind of division it comes from: `simétrica` (symmetric), `polar` (a filament releases a piece from one end), `fragmentación` (a filament breaks into several cells) |
| `completo` | born and divided within the movie (a complete cycle) |
| `fotos` | number of frames in which it is seen |

Values of `final`:

| Value | Meaning |
|---|---|
| `se divide` | divides |
| `sale o deja de verse` | leaves through the exit, or is lost from view near it |
| `desaparece en el canal (muere, se lisa o se pierde)` | disappears inside the channel (dies, lyses, or is lost by the detector) |
| `se pierde (hueco largo de fotos malas)` | its track is cut by a long gap without usable images (e.g. no images for a long time) |
| `fin de la película` | still there at the end of the movie |

A summary of the cycles that pass the usual quality filter (complete, symmetric birth, `reparto` 0.35–0.65,
`Ld/L0` 1.4–2.8) is shown in step 5 of the notebook, and all cells of all positions are written to
`Stacks/linaje_ciclos_todas.csv` (with a column `pos`).

`_linaje_<FLUO>.csv` (step 6):

| Column | Meaning |
|---|---|
| `canal, foto, celula` | as in `_linaje.csv` |
| `fluo_media` | mean intensity inside the cell (central columns, without 2 px at each pole) |
| `fondo_pared` | median intensity of the channel walls in the same rows and frame |
| `fondo_medio` | background of the medium (empty stretches of all channels of the position in that frame) |
| `fluo_neta` | `fluo_media − fondo_pared − fondo_medio`: proportional to the concentration |
| `fluo_total` | `fluo_neta × length`: proportional to the amount |
| `filas` | rows averaged |
| `toca_salida` | the box reaches the exit: the visible part may not be representative |

## `Stacks/Linaje/` (step 4)

`<base>-BF_linaje.tif`: RGB movie with one coloured, numbered box per cell (compressed TIFF: open it in Fiji
with *File › Import › Bio-Formats*).

## `Stacks/Excel/` (step 5)

`<base>_longitud_madres.xlsx`: mother length per channel and frame. The texts inside the sheet
(headers, notes, cell comments) are in Spanish: `hora` = time, `foto` = frame; the sheet `Notas` explains the
colours (red: beyond the exit; yellow: no reliable length; grey: bad frame; green: corrected by hand).
