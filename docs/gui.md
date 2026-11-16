# Graphical interface

For people who do not program: the whole pipeline from a window in the web browser, without typing any
command. It runs on your own computer (nothing is sent anywhere) and writes exactly the same files as the
programs and the notebook, so you can switch between them at any time.

## Installing it (once)

1. Download the pipeline: on the GitHub page press **Code › Download ZIP** and unzip it where you want to
   keep it (e.g. in Documents). Do not move the folder afterwards, or run the installer again if you do.
2. Double-click the installer in that folder:
   - **Windows:** `install_windows.bat`. If Windows shows *"Windows protected your PC"*, press
     *More info › Run anyway* (the file is not signed).
   - **macOS:** `install_mac.command`. The first time, macOS may refuse to open it because it was downloaded
     from the internet: right-click it › **Open** › **Open**. (On Linux, run `bash install_mac.command`.)
3. If conda is not installed, it offers to download and install **Miniforge** (a small, free Python
   distribution, ~100 MB) in your user folder. Then it creates the environment `mothermachine` with all the
   packages (about 10 minutes; it needs internet) and checks that everything works.

On Windows it also puts a **Mother machine** shortcut on the Desktop. Running the installer again updates the
packages (e.g. after downloading a new version of the pipeline).

If you prefer to install it by hand, follow [INSTALL.md](../INSTALL.md): the interface only needs the extra
package `streamlit`, which is in `requirements.txt`.

## Opening it

Double-click **Mother machine** on the Desktop (Windows) or `start_windows.bat` / `start_mac.command` in the
folder of the pipeline. A small window with messages opens, and after a few seconds the interface opens in
your web browser (at <http://localhost:8501>). **Keep the small window open** while you use the interface;
closing it stops the interface. You can close and reopen the browser tab at any time: a run keeps going and
you see its progress when you come back.

From the scripts folder you can also start it with `streamlit run interfaz.py`.

## Using it

**Panel on the left.** It has the pages of the interface (*Go to*: 1 · Images, 2 · Run, 3 · Check
alignment, 4 · Results, Help). Choose the folder with the images of one experiment (**📂 Choose the images folder…**,
or paste its path). The images must be named as the microscope saves them,
`<position>_<yymmdd>_<hhmmss>_anything.tif`. The results go to the folder `Stacks` next to it. Write the
names of the fluorescences, if there are any, from the brightest to the dimmest (e.g. `GFP, PI`).

**1 · Images.** The first time, the brightness of every image is measured (a progress bar shows it; about
1 s per 100 images, kept in `Stacks/interfaz/medianas.csv` for the next times). Number of images per position
and a plot of the brightness of every image over time:
bright-field images are the brightest, and the fluorescence ones are told apart by brightness.

- *Assign the fluorescence by time*: when two fluorescences are equally bright (e.g. GFP and PI), say which
  rounds were which: one row per round with its start and end time (e.g. PI from 16:36 to 16:46), and
  optionally that the last fluorescence image of each position is, e.g., PI. The table below shows how many
  images of each fluorescence each position gets. **Save this assignment** writes `Stacks/canal_imagenes.csv`.
- *Repeated images*: two bright-field images of the same position taken less than 4 minutes apart (a
  retaken photo, or a photo of another field of the chip). Tick the one to leave out.
- *Look at the images of a position*: thumbnails of every bright-field image; tick the ones to leave out
  (out of focus, another field of the chip, a bubble…). Their fluorescence images go with them. The list is
  `Stacks/imagenes_descartadas.csv`.

**2 · Run.** Choose the positions (none = all) and the steps, and press **▶ Start**. The bar shows the step
and position, and below are the messages of the programs. Steps already done are skipped, so after stopping
you just press Start again. **Stop after this position** finishes the current position first; **Stop now**
stops immediately (the position that was half done is redone next time).

- *Region used to align*: the chip numbers (usual). Choose **Whole image** when the stage drifts a lot and
  the numbers leave the image, with a maximum shift of 160 px.
- *Redo the steps*: redo everything even if it was already done (e.g. after changing a setting).

**3 · Check alignment.** One row per position with the alignment quality (1 = perfect; below 0.8, look at
it), the residual shift between consecutive frames (should be 0–1 px) and the rotation of the chip. For each
position, the frames that may be badly aligned are shown next to the reference frame: if one shows another
field of the chip or is out of focus, tick **Leave out this image** and save. Then run step 1 again (2 · Run):
only the positions with images left out are redone, followed by the next steps.

**4 · Results.** A table per position (cells, divisions, cells that disappear inside the channel, valid
cycles, cycle duration, lengths, growth rate, and how good the detection was), the growth rate of every
position over time, and a viewer of the stacks with one box per cell, frame by frame. Buttons open the files
in their usual program: the stacks in Fiji, the Excel files, the table of all cells.

## Correcting boxes

Boxes are corrected in Fiji with the macro `Cajas_Canales.ijm` ([training.md](training.md)). In 4 · Results,
**Open the straightened channels** opens the image the macro works on. After saving your corrections, run
step 3 again in 2 · Run: the positions you corrected are redone with your boxes, followed by steps 4–7.

## Files written by the interface

| File | Content |
|---|---|
| `Stacks/canal_imagenes.csv` | channel of each image (`archivo, canal`), when assigned by time |
| `Stacks/imagenes_descartadas.csv` | images left out (`archivo, motivo`) |
| `Stacks/interfaz/ajustes_interfaz.json` | the settings of the interface for this experiment |
| `Stacks/interfaz/medianas.csv` | brightness of every image, measured once |
| `Stacks/interfaz/ajustes.json`, `estado.json`, `registro.txt` | settings, progress and messages of the last run |
| `Stacks/interfaz/rehacer_stacks.txt`, `rehacer_pasos.json` | positions to redo in the next run |

The run itself is done by `scripts/ejecutar_pipeline.py`, which can also be used from the terminal with a
settings file (see the top of that file).

## If something goes wrong

- **After updating the pipeline**, close the small window and open the interface again, so that it uses the new
  programs.
- **The browser does not open:** open <http://localhost:8501> yourself while the small window is open.
- **"Port 8501 is already in use":** the interface is already open (maybe in another browser window).
- **A run stops with an error:** the messages are in 2 · Run and in `Stacks/interfaz/registro.txt`.
- **The installer fails:** check the internet connection and run it again; otherwise follow
  [INSTALL.md](../INSTALL.md).
