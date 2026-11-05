# Reviewing, annotating and retraining the detector

The cell detector is a small neural network (`scripts/modelos/detector_centros.onnx`, ~750 000
parameters). For every row of a straightened channel it predicts where the centre of a cell is and its
distance to both poles. It looks at each frame together with the two frames before and after it, so it
tells cells (which move and grow) from dirt and chip marks (which do not).

It was trained on ~5 000 hand-annotated channel frames of *E. coli* in bright field, from LB and from
several antibiotic treatments. On that kind of image it finds 97 % of the cells, and 98.5 % of its boxes
are correct. On images that look different it may need examples of yours.

## The Fiji macro `Cajas_Canales.ijm`

Install it as described in [INSTALL.md](../INSTALL.md#6-fiji-and-the-review-macro-recommended) and open a
straightened-channel image, `Stacks/Canales/<pos>_canales_BF.tif`. It has one slice per channel, with all
its frames side by side and the closed end at the top. Keyboard shortcuts:

| Key | Action |
|---|---|
| `1` | save the rectangle you drew as a box |
| `2` | delete the box under the mouse |
| `3` | update the selected box after moving its edges |
| `4` | copy the boxes of the previous annotated frame into this one |
| `5` | summary: annotated frames and boxes per channel |
| `6` | mark the frame under the mouse: empty channel, bad frame (this channel or all), or remove the mark |
| `7` | select the box under the mouse |
| `8` | propose the detector's boxes in the frame under the mouse (doubtful ones in yellow) |
| `9` | go to the next frame to review |
| `0` | help |

**Where the shortcuts get their data.**

- `8` reads the boxes written by `detectar_celulas.py predecir` (or `proponer`).
- `9` follows a review list:
  - `_revisar.txt`: doubtful detections, written by `predecir`;
  - `_revisar_seguimiento.txt`: frames where the tracking is least certain, written by
    `detectar_celulas.py revision`;
  - `_revisar_etiquetas.txt`: an optional list of your own boxes to double-check. No program in this
    repository writes it; any file with the same format (`channel frame reason` per line) works.

Everything is saved automatically in `<pos>_canales_BF_cajas.zip`, next to the image.

### What to draw

In each frame you annotate:

- one box per cell, from pole to pole;
- **all** cells from the closed end downwards, without skipping any, down to where you can no longer see
  them with confidence;
- nothing below that point.

Some special cases:

- A cell that goes past the exit: box it as far as you can see it. It is marked in red and not used for
  lengths.
- An empty channel: press `6` › *Empty channel*.
- Out-of-focus or shifted frames: press `6` › *Bad frame*. The tracking then skips them.

You do not need to annotate every frame: frames spread over time and over different situations
(crowded channels, filaments, dying cells, the first frames) are the most useful. The quickest way is to
press `8` and correct the proposal.

### Your corrections are used immediately

When you run `detectar_celulas.py predecir` again (or step 3 of the notebook), the frames you corrected use
your boxes instead of the network's, and the lineage is recomputed. To correct a few mistakes, this is
all you need: there is no need to retrain.

## Retraining the neural network

Needed only if the detector fails on many frames of a new kind of image. Install PyTorch first
(`pip install -r requirements-training.txt`).

1. **Annotate** some frames of several positions with the macro (a few hundred frames is a good start;
   the more varied, the better).
2. **Gather the training data.** Put in one folder, for every annotated position:
   - `<base>_canales_BF_cajas.zip`;
   - `<base>_canales_BF.tif` and `<base>_canales_BF_geometria.csv`;
   - if they exist, `<base>_canales_BF_borradas.txt`, `<base>_canales_BF_sin_datos.csv` and the folder's
     `fotos_validas.csv`.

   Keep some channels out as a test set by listing them in `conjunto_prueba.csv` (`imagen,canal`) in
   that folder. Then build the data file:
   ```bash
   cd scripts
   python centros_datos.py /path/to/annotated_folder datos.pkl
   ```
   Your boxes are the examples. Proposals you deleted count as "no cell here", with more weight. Rows
   below your last box are background, with half weight.
3. **Train** (default 14 epochs; ~40 min on a CPU for ~5 000 frames). The best model on the validation
   channels is saved:
   ```bash
   python entrenar_centros.py datos.pkl my_detector.pt 14
   ```
4. **Export it** to ONNX and install it, keeping a copy of the previous model:
   ```bash
   cp modelos/detector_centros.onnx modelos/detector_centros_old.onnx
   python -c "import detector_centros as D; D.exportar_onnx('my_detector.pt', 'modelos/detector_centros.onnx')"
   cp my_detector.pt modelos/detector_centros.pt
   ```
5. Run step 3 again. Positions are redone automatically because the network is newer than their lineage.

To combine your annotations with the ones the shipped model was trained on, you would need the original
annotated images, which are not part of this repository. Training on your annotations alone works if they
cover the situations you care about.

## The tree-based detector (fallback)

`modelos/detector_canales.joblib` is an older gradient-boosting detector, used only when onnxruntime is
not installed. It can be retrained with `python detectar_celulas.py entrenar <folder>/*_cajas.zip`. Use
`validar` (cross-validation by channel) and `probar` (on the channels of `conjunto_prueba.csv`) to
evaluate it.
