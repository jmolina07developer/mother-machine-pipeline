// ==========================================================================
//  Cajas_Canales.ijm
//  Fiji macro set for drawing a box (rectangle) around each cell
//  in the straightened channels produced by canales_enderezados.py
//  (<folder>/Canales/<base>_canales_BF.tif).
//
//  The image: one slice per channel. In each slice, all the frames of the channel side
//  by side (frame 1 on the left; its number is at the top), separated by a
//  black strip. The closed end is at the top; the yellow lines are the closed end and
//  the exit of the channel.
//
//  Install: Plugins › Macros › Install…  (choose this file)
//
//  WHAT TO DRAW (rectangle tool):
//    - in each frame you annotate, one box per cell covering it ENTIRELY (pole to
//      pole; the width does not matter much, but keep it inside the frame);
//    - ALL cells from the closed end downwards, without skipping any, down to
//      where you can no longer see them with confidence; draw nothing below that;
//    - a cell that goes past the exit: box it as far as you can see it (even if that
//      is below the exit). It is marked in red automatically and is not used later to
//      measure lengths, but it is used to know where the cell is.
//    - you do not need to annotate every frame: a frame with at least one box counts
//      as annotated; the others are not used. Frames spread out over time are best.
//    - a frame in which the channel is empty: [6] with the mouse over it (Enter).
//    - a bad frame (out of focus, shifted, bubble…): [6] and choose "Bad
//      frame" (this channel only or all channels). It is not used for training or
//      measuring: the tracking skips it.
//
//  Keyboard shortcuts (with the channel window in front):
//    [1] Save box          -> saves the drawn rectangle
//    [2] Delete            -> deletes the box under the mouse (if it was a detector
//                             proposal, it is logged in <image>_borradas.txt: there is
//                             no cell there, and the detector learns it when retrained)
//    [3] Update box        -> click on a box (it gets selected), adjust its edges and
//                             press [3]; no need to touch the ROI Manager
//    [7] Select box        -> selects the box under the mouse (if clicking does not
//                             select it)
//    [4] Copy boxes        -> copies the boxes of the previous annotated frame into
//                             the frame under the mouse (then adjust them with [3])
//    [5] Summary           -> annotated frames and boxes per channel (in the Log)
//    [6] Mark frame        -> the frame under the mouse: empty channel, bad frame (in
//                             this channel or in all of them) or remove the mark
//    [8] Propose boxes     -> puts the centre-based detector's boxes (neural network;
//                             detectar_celulas.py predecir or proponer) in the frame
//                             under the mouse; check them before moving on: the frame
//                             counts as annotated just as if you had drawn them. In
//                             yellow, the uncertain ones (confidence < 0.6: check them more carefully)
//    [9] Next to review    -> goes to the next frame that the detector flags as
//                             uncertain (<image>_revisar.txt) and selects it. If
//                             <image>_revisar_seguimiento.txt (where the tracking is
//                             least certain: detectar_celulas.py revision) or
//                             <image>_revisar_etiquetas.txt (audit of your boxes) also
//                             exists, the first time it asks which list to follow
//    [0] Help
//
//  Everything is saved automatically in <image>_cajas.zip, next to the image:
//    C05_f023_c02     box no. 2 of channel 5 in frame 23
//    C05_f023_VACIA   channel 5 is empty in frame 23
//    C05_f023_MALA    frame 23 of channel 5 is bad (C00_f023_MALA: in all channels)
//  leer_cajas.py then converts it into a table (channel, frame, top and bottom
//  pole in kymograph rows, whether the cell exits the channel…).
// ==========================================================================

var imgActiva = "";
var rutaZip = "";
var ajenoEjemplo = "";
var COLOR_OK = "#00ff66";
var COLOR_SALE = "#ff3030";
var COLOR_DUDOSA = "#ffd000";
var DUDOSA = false;
var REV_I = -1;
var REV_LISTA = "";      // list followed by [9] in this image
var REV_IMG = "";
var GEO_ANCHO = NaN;
var GEO_SEP = NaN;
var GEO_FOTOS = NaN;
var GEO_SALIDA = newArray(1000);
var BX = newArray(0);
var BY = newArray(0);
var BW = newArray(0);
var BH = newArray(0);
var BI = newArray(0);
var ORIGEN = "manual";
var rutaDetector = "";
var rutaBorradas = "";

// ==========================================================================
macro "Save box [1]" {
    preparar();
    if (selectionType() != 0)
        exit("Draw a rectangle that covers ONE whole cell (in a single frame) and press [1].");
    getSelectionBounds(x, y, w, h);
    ORIGEN = "manual";
    guardarCaja(x, y, w, h, -1, true);
}

// ==========================================================================
macro "Delete [2]" {
    preparar();
    // 1) the box under the mouse (the usual case); 2) the selected one; 3) the last one
    getCursorLoc(xc, yc, zc, flags);
    i = cajaEn(xc, yc);
    pedir = false;
    if (i < 0) i = roiManager("index");
    if (i < 0) { i = roiManager("count") - 1; pedir = true; }
    if (i < 0) exit("There are no boxes.");
    nom = call("ij.plugin.frame.RoiManager.getName", "" + i);
    if (pedir) if (!getBoolean("There is no box under the mouse.\nDelete the last one (" + nom + ")?")) exit;
    roiManager("select", i);
    // a detector-proposed box that you delete: there is no cell there (logged for training)
    if (Roi.getProperty("origen") == "detector" && indexOf(nom, "_c") == 8) {
        getSelectionBounds(bx, by, bw, bh);
        File.append(substring(nom, 1, 3) + " " + substring(nom, 5, 8) + " " + d2s(by, 2) + " " + d2s(by + bh, 2) + " " + replace(ahora(), " ", "_"), rutaBorradas);
    }
    roiManager("delete"); roiManager("deselect");
    run("Select None");
    print("Deleted: " + nom);
    guardarZip();
}

// ==========================================================================
macro "Update box [3]" {
    preparar();
    if (selectionType() != 0)
        exit("Click on the box (it gets selected), adjust its edges and press [3].");
    getSelectionBounds(x, y, w, h);
    // the box being edited: the one in the same frame that overlaps the rectangle the most
    i = cajaSolapada(x, y, w, h);
    if (i < 0) exit("There is no box there. For a new box use [1].");
    ORIGEN = "manual";
    guardarCaja(x, y, w, h, i, false);
}

// ==========================================================================
macro "Select box [7]" {
    preparar();
    getCursorLoc(xc, yc, zc, flags);
    i = cajaEn(xc, yc);
    if (i < 0) exit("Put the mouse over a box and press [7].");
    roiManager("select", i);
    showStatus("Selected " + call("ij.plugin.frame.RoiManager.getName", "" + i) + ": adjust it and press [3]");
}

// ==========================================================================
macro "Copy boxes [4]" {
    preparar();
    e = getInfo("slice.label");
    c = canalDe(e); paso = GEO_ANCHO + GEO_SEP;
    getCursorLoc(xc, yc, zc, flags);
    destino = floor(xc / paso) + 1;
    if (destino < 2 || destino > GEO_FOTOS) exit("Put the mouse over the frame you want to copy the boxes into and press [4].");
    if (cajasDe(c, destino, false) > 0)
        if (!getBoolean("Frame " + destino + " already has boxes. Add the copied ones anyway?")) exit;
    origen = -1;
    for (f = destino - 1; f >= 1 && origen < 0; f--) if (cajasDe(c, f, false) > 0) origen = f;
    if (origen < 0) exit("There is no annotated frame to the left of frame " + destino + " in this channel.");
    nb = cajasDe(c, origen, true);                          // fills BX, BY, BW, BH
    bx = Array.copy(BX); by = Array.copy(BY); bw = Array.copy(BW); bh = Array.copy(BH);
    dx = (destino - origen) * paso;
    ORIGEN = "copiada";
    for (k = 0; k < nb; k++) guardarCaja(bx[k] + dx, by[k], bw[k], bh[k], -1, false);
    ORIGEN = "manual";
    print("Copied " + nb + " boxes from frame " + origen + " to frame " + destino + ": adjust them ([3]) or delete them ([2]).");
}

// ==========================================================================
macro "Summary [5]" {
    preparar();
    nc = nSlices;
    cajas = newArray(nc + 1); fotos = newArray(nc + 1); vacias = newArray(nc + 1); malas = newArray(nc + 1);
    malasTodas = 0;
    vistas = newArray(nc + 1);                             // frames already counted, as text
    for (c = 1; c <= nc; c++) vistas[c] = ",";
    n = roiManager("count");
    for (i = 0; i < n; i++) {
        nom = call("ij.plugin.frame.RoiManager.getName", "" + i);
        if (startsWith(nom, "C") && indexOf(nom, "_f") == 3) {
            c = parseInt(substring(nom, 1, 3));
            f = parseInt(substring(nom, 5, 8));
            if (c == 0 && endsWith(nom, "_MALA")) malasTodas++;
            else if (c >= 1 && c <= nc && endsWith(nom, "_MALA")) malas[c]++;
            else if (c >= 1 && c <= nc) {
                if (endsWith(nom, "_VACIA")) vacias[c]++;
                else cajas[c]++;
                if (indexOf(vistas[c], "," + f + ",") < 0) { vistas[c] = vistas[c] + f + ","; fotos[c]++; }
            }
        }
    }
    print("\n===== " + imgActiva + " =====");
    tc = 0; tf = 0;
    for (c = 1; c <= nc; c++) {
        if (fotos[c] > 0) {
            t = "channel " + IJ.pad(c, 2) + ": " + fotos[c] + " annotated frames, " + cajas[c] + " boxes";
            if (vacias[c] > 0) t = t + ", " + vacias[c] + " empty";
            print(t);
            tc += cajas[c]; tf += fotos[c];
        }
    }
    for (c = 1; c <= nc; c++) if (malas[c] > 0) print("channel " + IJ.pad(c, 2) + ": " + malas[c] + " bad frames");
    if (malasTodas > 0) print(malasTodas + " bad frames in all channels");
    print("Total: " + tf + " annotated frames, " + tc + " boxes");
    print("File: " + rutaZip);
}

// ==========================================================================
macro "Mark frame [6]" {
    preparar();
    e = getInfo("slice.label");
    c = canalDe(e); paso = GEO_ANCHO + GEO_SEP;
    getCursorLoc(xc, yc, zc, flags);
    f = floor(xc / paso) + 1;
    if (f < 1 || f > GEO_FOTOS) exit("Put the mouse over the frame and press [6].");
    nVacia = "C" + IJ.pad(c, 2) + "_f" + IJ.pad(f, 3) + "_VACIA";
    nMala = "C" + IJ.pad(c, 2) + "_f" + IJ.pad(f, 3) + "_MALA";
    nTodas = "C00_f" + IJ.pad(f, 3) + "_MALA";
    marcas = "";
    if (indiceDe(nVacia) >= 0) marcas = marcas + " empty channel;";
    if (indiceDe(nMala) >= 0) marcas = marcas + " bad in this channel;";
    if (indiceDe(nTodas) >= 0) marcas = marcas + " bad in all channels;";
    ops = newArray("Empty channel (no cells)", "Bad frame (this channel only)", "Bad frame (all channels)", "Remove the mark");
    porDefecto = ops[0];
    if (marcas != "") porDefecto = ops[3];
    Dialog.create("Mark frame");
    t = "Channel " + c + ", frame " + f;
    if (marcas != "") t = t + "\nAlready marked:" + marcas;
    Dialog.addMessage(t);
    Dialog.addRadioButtonGroup("", ops, 4, 1, porDefecto);
    Dialog.show();
    op = Dialog.getRadioButton();
    if (op == ops[3]) {
        n = 0;
        nombres = newArray(nVacia, nMala, nTodas);
        for (k = 0; k < 3; k++) {
            i = indiceDe(nombres[k]);
            if (i >= 0) { roiManager("select", i); roiManager("delete"); n++; }
        }
        roiManager("deselect"); run("Select None");
        if (n == 0) exit("Frame " + f + " had no mark.");
        print("Channel " + c + ", frame " + f + ": mark removed");
        guardarZip();
        exit;
    }
    if (cajasDe(c, f, false) > 0) exit("Frame " + f + " has boxes: delete them first ([2]).");
    s = getSliceNumber();
    if (op == ops[0]) {
        if (indiceDe(nVacia) >= 0) exit("It is already marked as an empty channel.");
        makePoint((f - 1) * paso + GEO_ANCHO / 2, 5, "medium cross");
        Roi.setName(nVacia); Roi.setPosition(s); Roi.setStrokeColor("#00b0ff");
        Roi.setProperty("tipo", "vacia"); nombre = nVacia; pos = s;
        texto = "empty channel in frame " + f;
    } else {
        todas = (op == ops[2]);
        nombre = nMala; pos = s;
        if (todas) { nombre = nTodas; pos = 0; }
        if (indiceDe(nombre) >= 0) exit("It is already marked that way.");
        makePoint((f - 1) * paso + GEO_ANCHO / 2, 15, "large circle");
        Roi.setName(nombre); Roi.setPosition(pos); Roi.setStrokeColor("#ff8c00");
        Roi.setProperty("tipo", "mala");
        texto = "frame " + f + " marked bad";
        if (todas) texto = texto + " in all channels"; else texto = texto + " in this channel";
    }
    Roi.setProperty("canal", "" + c);
    Roi.setProperty("foto", "" + f);
    Roi.setProperty("imagen", imgActiva);
    Roi.setProperty("anotado_el", ahora());
    anadirAlManager(nombre, pos);
    print(nombre + "   " + texto);
    guardarZip();
}

// ==========================================================================
macro "Propose boxes [8]" {
    preparar();
    e = getInfo("slice.label");
    c = canalDe(e); paso = GEO_ANCHO + GEO_SEP;
    getCursorLoc(xc, yc, zc, flags);
    f = floor(xc / paso) + 1;
    if (f < 1 || f > GEO_FOTOS) exit("Put the mouse over the frame and press [8].");
    if (!File.exists(rutaDetector)) exit("There are no detector boxes for this image\n(missing " + File.getName(rutaDetector) + ").");
    if (cajasDe(c, f, false) > 0)
        if (!getBoolean("Frame " + f + " already has boxes. Add the detector ones too?")) exit;
    lineas = split(File.openAsString(rutaDetector), "\n");
    x0 = (f - 1) * paso;
    n = 0; nd = 0;
    ORIGEN = "detector";
    for (i = 0; i < lineas.length; i++) {
        p = split(lineas[i], " ");
        if (p.length >= 4) {
            if (parseInt(p[0]) == c && parseInt(p[1]) == f) {
                y0 = parseFloat(p[2]); y1 = parseFloat(p[3]);
                DUDOSA = false;
                if (p.length >= 6) DUDOSA = (parseInt(p[5]) == 1);
                if (y1 - y0 >= 3) { guardarCaja(x0 + 2, y0, GEO_ANCHO - 4, y1 - y0, -1, false); n++; if (DUDOSA) nd++; }
            }
        }
    }
    DUDOSA = false;
    ORIGEN = "manual";
    t = "Frame " + f + ": " + n + " detector boxes";
    if (nd > 0) t = t + " (" + nd + " uncertain, in yellow)";
    print(t + ". Check them: correct with [3], delete with [2] (the box under the mouse),");
    print("add any missing ones with [1] and delete those below the point where you can no longer see with confidence.");
}

// ==========================================================================
macro "Next to review [9]" {
    preparar();
    ruta = replace(rutaDetector, "_detector.txt", "_revisar.txt");
    rutaEt = replace(rutaDetector, "_detector.txt", "_revisar_etiquetas.txt");
    rutaSeg = replace(rutaDetector, "_detector.txt", "_revisar_seguimiento.txt");
    if (REV_IMG != getTitle()) { REV_LISTA = ""; REV_IMG = getTitle(); REV_I = -1; }
    if ((File.exists(rutaEt) || File.exists(rutaSeg)) && REV_LISTA == "") {
        opciones = newArray(0);
        if (File.exists(rutaSeg)) opciones = Array.concat(opciones, "Uncertain tracking (_revisar_seguimiento.txt)");
        if (File.exists(rutaEt)) opciones = Array.concat(opciones, "Audit of your boxes (_revisar_etiquetas.txt)");
        opciones = Array.concat(opciones, "Uncertain detections (_revisar.txt)");
        Dialog.create("Next to review [9]");
        Dialog.addChoice("List:", opciones);
        Dialog.show();
        eleccion = Dialog.getChoice();
        REV_LISTA = "detector";
        if (startsWith(eleccion, "Audit of your boxes")) REV_LISTA = "etiquetas";
        if (startsWith(eleccion, "Uncertain tracking")) REV_LISTA = "seguimiento";
        REV_I = -1;
    }
    if (REV_LISTA == "etiquetas") ruta = rutaEt;
    if (REV_LISTA == "seguimiento") ruta = rutaSeg;
    if (!File.exists(ruta)) exit("There is no list of frames to review\n(missing " + File.getName(ruta) + "; run detectar_celulas.py predecir again).");
    lineas = split(File.openAsString(ruta), "\n");
    if (lineas.length == 0) exit("There is nothing to review.");
    // start after the frame under the mouse (or after the last one visited)
    e = getInfo("slice.label"); cAct = canalDe(e); paso = GEO_ANCHO + GEO_SEP;
    getCursorLoc(xc, yc, zc, flags);
    fAct = floor(xc / paso) + 1;
    sig = -1;
    if (REV_I >= 0 && REV_I < lineas.length) sig = REV_I + 1;
    else {
        for (i = 0; i < lineas.length && sig < 0; i++) {
            p = split(lineas[i], " ");
            if (p.length >= 2) if (parseInt(p[0]) > cAct || (parseInt(p[0]) == cAct && parseInt(p[1]) > fAct)) sig = i;
        }
    }
    if (sig < 0 || sig >= lineas.length) { REV_I = -1; exit("No more frames to review (press [9] again to start over)."); }
    REV_I = sig;
    p = split(lineas[sig], " ");
    c = parseInt(p[0]); f = parseInt(p[1]);
    motivo = substring(lineas[sig], indexOf(lineas[sig], p[2]));
    if (c != cAct) {
        encontrado = false;
        for (s = 1; s <= nSlices && !encontrado; s++) {
            setSlice(s);
            if (canalDe(getInfo("slice.label")) == c) encontrado = true;
        }
        if (!encontrado) exit("Cannot find channel " + c + " in this image.");
    }
    makeOval((f - 1) * paso - 2, 0, GEO_ANCHO + 4, getHeight());      // oval (not rectangle): [1] and [3] do not take it for a box
    print("Review (" + (sig + 1) + "/" + lineas.length + "): channel " + c + ", frame " + f + ": " + motivo);
    showStatus("Channel " + c + ", frame " + f + ": " + motivo + "  ([8] to propose boxes, [9] next)");
}

// ==========================================================================
macro "Help [0]" {
    print("\n===== Boxes in channels =====");
    print("[1] Save box   [2] Delete (the box under the mouse)   [3] Update box");
    print("[4] Copy boxes from the previous frame   [5] Summary   [6] Mark frame (empty / bad)   [7] Select box");
    print("[8] Propose the boxes of the centre-based detector in the frame under the mouse (then check them; uncertain ones in yellow)");
    print("[9] Go to the next frame to review: uncertain detections, uncertain tracking (_revisar_seguimiento.txt) or the audit of your boxes (_revisar_etiquetas.txt)");
    print("To correct a box: click inside it (with the rectangle tool), move its edges and press [3].");
    print("In each frame you annotate: one box per cell, all of them from the closed end downwards to");
    print("where you can no longer see them, without skipping any. Those that go past the exit are shown in red and");
    print("are not used to measure lengths.");
}

// ==========================================================================
//  Helper functions
// ==========================================================================

function guardarCaja(x, y, w, h, iRep, comprobar) {
    e = getInfo("slice.label");
    c = canalDe(e);
    ancho = GEO_ANCHO; sep = GEO_SEP; salida = GEO_SALIDA[c]; nf = GEO_FOTOS;
    if (salida <= 0) exit("Cannot find the exit of channel " + c + " in the _macro.txt file.");
    paso = ancho + sep;
    tile = floor((x + w / 2) / paso);
    if (tile < 0 || tile >= nf) exit("The box is outside the frames.");
    x0 = tile * paso;
    xa = maxOf(x, x0); xb = minOf(x + w, x0 + ancho);
    if (xb - xa < 2) exit("The box must be inside a frame (between the black strips).");
    if (y < 0) { h = h + y; y = 0; }
    if (y + h > getHeight()) h = getHeight() - y;
    if (h < 3) exit("The box is too small.");
    foto = tile + 1;
    s = getSliceNumber();
    filaSup = y - 0.5; filaInf = y + h - 0.5;
    sale = (filaInf >= salida - 1);
    if (comprobar) {                                        // does it overlap a lot with another box?
        nb = cajasDe(c, foto, true);
        for (k = 0; k < nb; k++) {
            sol = minOf(y + h, BY[k] + BH[k]) - maxOf(y, BY[k]);
            if (sol > 0.5 * minOf(h, BH[k]))
                if (!getBoolean("This box overlaps a lot with another one in the same frame.\nSave it anyway?")) { makeRectangle(x, y, w, h); exit; }
        }
    }
    nombre = "";
    if (iRep >= 0) {
        viejo = call("ij.plugin.frame.RoiManager.getName", "" + iRep);
        if (startsWith(viejo, "C" + IJ.pad(c, 2) + "_f" + IJ.pad(foto, 3) + "_c")) nombre = viejo;
        roiManager("select", iRep); roiManager("delete"); roiManager("deselect");
    }
    if (nombre == "") nombre = "C" + IJ.pad(c, 2) + "_f" + IJ.pad(foto, 3) + "_c" + IJ.pad(siguienteCaja(c, foto), 2);
    makeRectangle(xa, y, xb - xa, h);
    Roi.setName(nombre);
    Roi.setPosition(s);
    if (sale) Roi.setStrokeColor(COLOR_SALE);
    else if (DUDOSA && ORIGEN == "detector") Roi.setStrokeColor(COLOR_DUDOSA);
    else Roi.setStrokeColor(COLOR_OK);
    Roi.setProperty("canal", "" + c);
    Roi.setProperty("foto", "" + foto);
    Roi.setProperty("tipo", "caja");
    Roi.setProperty("origen", ORIGEN);
    if (DUDOSA && ORIGEN == "detector") Roi.setProperty("dudosa", "1");
    Roi.setProperty("fila_sup", d2s(filaSup, 2));
    Roi.setProperty("fila_inf", d2s(filaInf, 2));
    Roi.setProperty("sale_del_canal", "" + sale);
    Roi.setProperty("imagen", imgActiva);
    Roi.setProperty("anotado_el", ahora());
    anadirAlManager(nombre, s);
    t = nombre + "   frame " + foto + ", rows " + d2s(filaSup, 1) + "-" + d2s(filaInf, 1);
    if (sale) t = t + "   (goes past the exit: will not be used for measuring)";
    print(t);
    guardarZip();
}

// Boxes of one channel and one frame; if guardar, stores their bounds in BX, BY, BW, BH
function cajasDe(c, f, guardar) {
    pref = "C" + IJ.pad(c, 2) + "_f" + IJ.pad(f, 3) + "_c";
    n = roiManager("count"); k = 0;
    idx = newArray(n);
    for (i = 0; i < n; i++)
        if (startsWith(call("ij.plugin.frame.RoiManager.getName", "" + i), pref)) { idx[k] = i; k++; }
    if (!guardar) return k;
    BX = newArray(k); BY = newArray(k); BW = newArray(k); BH = newArray(k); BI = newArray(k);
    if (k == 0) return 0;
    roiManager("Associate", "false");
    for (j = 0; j < k; j++) {
        roiManager("select", idx[j]);
        getSelectionBounds(a, b, cw, ch);
        BX[j] = a; BY[j] = b; BW[j] = cw; BH[j] = ch; BI[j] = idx[j];
    }
    roiManager("deselect");
    run("Select None");
    roiManager("Associate", "true");
    return k;
}

// Index (ROI Manager) of the box containing the point (x, y) in the current channel, or -1
function cajaEn(x, y) {
    e = getInfo("slice.label");
    c = canalDe(e);
    paso = GEO_ANCHO + GEO_SEP;
    f = floor(x / paso) + 1;
    if (f < 1 || f > GEO_FOTOS) return -1;
    nb = cajasDe(c, f, true);
    mejor = -1; hmin = 1e9;
    for (k = 0; k < nb; k++) {
        if (x >= BX[k] - 1 && x <= BX[k] + BW[k] + 1 && y >= BY[k] - 1 && y <= BY[k] + BH[k] + 1 && BH[k] < hmin) {
            mejor = BI[k]; hmin = BH[k];                    // if there are several, the smallest one
        }
    }
    return mejor;
}

// Index of the box in the same frame that overlaps the rectangle the most (vertically), or -1
function cajaSolapada(x, y, w, h) {
    e = getInfo("slice.label");
    c = canalDe(e);
    paso = GEO_ANCHO + GEO_SEP;
    f = floor((x + w / 2) / paso) + 1;
    if (f < 1 || f > GEO_FOTOS) return -1;
    nb = cajasDe(c, f, true);
    mejor = -1; smax = 0;
    for (k = 0; k < nb; k++) {
        sol = minOf(y + h, BY[k] + BH[k]) - maxOf(y, BY[k]);
        if (sol > smax) { smax = sol; mejor = BI[k]; }
    }
    return mejor;
}

function siguienteCaja(c, f) {
    pref = "C" + IJ.pad(c, 2) + "_f" + IJ.pad(f, 3) + "_c";
    mx = 0;
    n = roiManager("count");
    for (i = 0; i < n; i++) {
        nom = call("ij.plugin.frame.RoiManager.getName", "" + i);
        if (startsWith(nom, pref)) {
            v = parseInt(substring(nom, lengthOf(pref)));
            if (!isNaN(v) && v > mx) mx = v;
        }
    }
    return mx + 1;
}

// Channel of the slice: the label starts with "canal 05"
function canalDe(e) {
    if (!startsWith(e, "canal ")) exit("This is not a straightened-channel image (the slice label is missing).");
    return parseInt(substring(e, 6, 8));
}

// Image geometry (written by canales_enderezados.py to <image>_macro.txt):
//   ancho 33 / sep 3 / fotos 73 / salida <channel> <row>
// (not read from the slice label because Fiji truncates it to 60 characters)
function cargarGeometria(ruta) {
    if (!File.exists(ruta)) exit("Cannot find " + File.getName(ruta) + " next to the image.\nRun canales_enderezados.py again for this position.");
    lineas = split(File.openAsString(ruta), "\n");
    GEO_SALIDA = newArray(1000);
    GEO_ANCHO = NaN; GEO_SEP = NaN; GEO_FOTOS = NaN;
    for (i = 0; i < lineas.length; i++) {
        p = split(String.trim(lineas[i]), " ");
        if (p.length < 2) continue;
        if (p[0] == "ancho") GEO_ANCHO = parseFloat(p[1]);
        else if (p[0] == "sep") GEO_SEP = parseFloat(p[1]);
        else if (p[0] == "fotos") GEO_FOTOS = parseFloat(p[1]);
        else if (p[0] == "salida" && p.length >= 3) GEO_SALIDA[parseInt(p[1])] = parseFloat(p[2]);
    }
    if (isNaN(GEO_ANCHO) || isNaN(GEO_SEP) || isNaN(GEO_FOTOS)) exit("The file " + File.getName(ruta) + " is incomplete.");
}

function preparar() {
    if (nImages == 0) exit("First open a channel image (<base>_canales_BF.tif).");
    t = getTitle();
    if (indexOf(t, "_canales") < 0) exit("This window is not a straightened-channel image (<base>_canales_BF.tif).");
    roiManager("Associate", "true");
    roiManager("UseNames", "true");
    if (t == imgActiva) return;
    if (imgActiva != "" && roiManager("count") > 0) {     // another image: its boxes are saved and we switch, without asking
        guardarZip();
        print("Saved the boxes of " + imgActiva + "; now annotating " + t);
        roiManager("reset");
    }
    REV_LISTA = ""; REV_IMG = "";
    if (imgActiva == "" && roiManager("count") > 0) {
        if (!getBoolean("The ROI Manager has " + roiManager("count") + " ROIs from before.\nClear it to annotate \"" + t + "\"?")) exit;
        roiManager("reset");
    }
    dir = getDirectory("image");
    if (dir == "") exit("Save the image to disk first.");
    base = t;
    punto = lastIndexOf(base, ".");
    if (punto > 0) base = substring(base, 0, punto);
    cargarGeometria(dir + base + "_macro.txt");
    imgActiva = t;
    REV_I = -1;
    rutaZip = dir + base + "_cajas.zip";
    rutaDetector = dir + base + "_detector.txt";
    rutaBorradas = dir + base + "_borradas.txt";
    if (File.exists(rutaZip)) {
        roiManager("open", rutaZip);
        print("Loaded " + roiManager("count") + " boxes from " + rutaZip);
        n = contarAjenos();
        if (n > 0) {
            roiManager("reset");
            imgActiva = ""; rutaZip = "";
            exit("Warning: " + base + "_cajas.zip has " + n + " boxes from ANOTHER image (" + ajenoEjemplo + ").\nNot loading it, to avoid mixing positions.");
        }
    } else {
        print("New box file: " + rutaZip);
    }
    roiManager("Show All without labels");
}

function guardarZip() {
    if (rutaZip == "") return;
    roiManager("deselect");
    if (roiManager("count") == 0) return;
    n = contarAjenos();
    if (n > 0) {
        showMessage("Boxes not saved", "The ROI Manager contains " + n + " ROIs from ANOTHER image (" + ajenoEjemplo + ").\nNot saving them to " + File.getName(rutaZip) + ", to avoid mixing positions.\nClear the ROI Manager and reopen only the image you want to annotate.\nDo not drag .zip files into Fiji: the macro loads the boxes of each image by itself.");
        exit;
    }
    roiManager("save", rutaZip);
}

// ROIs whose "imagen" property is not the active image
function contarAjenos() {
    n = roiManager("count"); ajenos = 0; ajenoEjemplo = "";
    roiManager("Associate", "false");
    for (i = 0; i < n; i++) {
        roiManager("select", i);
        k = Roi.getProperty("imagen");
        if (k != "" && k != imgActiva) {
            ajenos++;
            if (ajenoEjemplo == "") ajenoEjemplo = k;
        }
    }
    roiManager("deselect");
    run("Select None");
    roiManager("Associate", "true");
    return ajenos;
}

function indiceDe(nombre) {
    n = roiManager("count");
    for (i = 0; i < n; i++) {
        if (call("ij.plugin.frame.RoiManager.getName", "" + i) == nombre) return i;
    }
    return -1;
}

function anadirAlManager(nombre, s) {
    roiManager("add");
    i = roiManager("count") - 1;
    roiManager("select", i);
    roiManager("rename", nombre);
    RoiManager.setPosition(s);
    roiManager("deselect");
    run("Select None");
}

function ahora() {
    getDateAndTime(an, mes, dsem, dia, h, m, seg, ms);
    return "" + an + "-" + IJ.pad(mes + 1, 2) + "-" + IJ.pad(dia, 2) + " " + IJ.pad(h, 2) + ":" + IJ.pad(m, 2) + ":" + IJ.pad(seg, 2);
}
