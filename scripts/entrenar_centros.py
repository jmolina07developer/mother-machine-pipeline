"""Train the centre-based detector. Usage: entrenar_centros.py datos.pkl salida.pt [epocas]"""
import sys, time, pickle, math
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import numpy as np, torch
import detector_centros as DCe

torch.set_num_threads(2); torch.manual_seed(0); rng = np.random.default_rng(0)
D = pickle.load(open(sys.argv[1], 'rb')); salida = sys.argv[2]; EP = int(sys.argv[3]) if len(sys.argv) > 3 else 14
CAN, FOT, LARGO = D['canales'], D['fotos'], D['LARGO']

# validation: 10 % of the non-test channels (whole channels)
claves = sorted({(f['base'], f['canal']) for f in FOT if not f['prueba']})
val = set(claves[i] for i in rng.permutation(len(claves))[:max(1, len(claves) // 10)])
tr = [f for f in FOT if not f['prueba'] and (f['base'], f['canal']) not in val]
va = [f for f in FOT if not f['prueba'] and (f['base'], f['canal']) in val]
print(f'training {len(tr)} images, validation {len(va)} images ({len(val)} channels)', flush=True)


def preparar(fotos):
    X = np.zeros((len(fotos), DCe.CIN, LARGO, D['ANCHO']), np.float16)
    C = np.zeros((len(fotos), LARGO), np.float32); DI = np.zeros((len(fotos), 2, LARGO), np.float32)
    M = np.zeros((len(fotos), LARGO), np.float32); Y = np.zeros((len(fotos), LARGO), np.int64); W = np.zeros((len(fotos), LARGO), np.float32)
    for i, f in enumerate(fotos):
        c = CAN[(f['base'], f['canal'])]; L = f['L']
        X[i] = DCe.entrada(c['tiles'], f['foto'] - 1, c['fila_fondo'], c['fila_salida'])
        C[i, :L], DI[i, :, :L], M[i, :L] = DCe.objetivos(f['cajas'], L)
        Y[i, :L] = f['y']; W[i, :L] = f['w']
    return X, C, DI, M, Y, W


t0 = time.time()
Xtr, Ctr, Dtr, Mtr, Ytr, Wtr = preparar(tr); Xva, Cva, Dva, Mva, Yva, Wva = preparar(va)
print(f'data ready {time.time() - t0:.0f} s, {Xtr.nbytes / 1e9:.2f} GB', flush=True)
cnt = np.array([(Wtr * (Ytr == k)).sum() for k in range(4)])
pc = torch.tensor(np.sqrt(cnt.sum() / (4 * np.maximum(cnt, 1))), dtype=torch.float32)

red = DCe.RedCentros(); npar = sum(p.numel() for p in red.parameters()); print('parameters', npar, flush=True)
opt = torch.optim.AdamW(red.parameters(), lr=2e-3, weight_decay=1e-4)
B = 16; pasos = EP * math.ceil(len(tr) / B)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=pasos, pct_start=0.1)


def aumentar(x):
    x = x.astype(np.float32)
    nf = 2 * DCe.VECINAS + 1
    for b in range(len(x)):
        x[b, :nf] = x[b, :nf] * rng.uniform(0.8, 1.25) + rng.normal(0, 0.15)
        x[b, :nf] += rng.normal(0, 0.05, x[b, :nf].shape)
        if rng.random() < 0.15:                      # no neighbours (start or end of the movie)
            x[b, :nf] = x[b, DCe.VECINAS]
        if rng.random() < 0.5: x[b] = x[b, :, :, ::-1]
    return torch.from_numpy(np.ascontiguousarray(x))


def evaluar():
    red.eval(); tot = 0; ok = 0; npred = 0; nver = 0; L = 0.0
    with torch.no_grad():
        for i in range(0, len(va), 64):
            x = torch.from_numpy(Xva[i:i + 64].astype(np.float32)); out = red(x)
            obj = [torch.from_numpy(a[i:i + 64]) for a in (Cva, Dva, Mva, Yva, Wva)]
            L += DCe.perdida(out, obj, pc)[0].item() * len(x)
            ce = torch.sigmoid(out[0]).numpy(); di = out[1].numpy()
            for j in range(len(x)):
                f = va[i + j]; pr = DCe.cajas_de(ce[j], di[j], L=f['L'])
                ver = f['cajas']; zona = max([b for _, b in ver]) + 3 if ver else 0
                pr = [(a, b) for a, b, _ in pr if a < zona]
                usados = set()
                for a, b in ver:
                    m = [k for k, (c, d) in enumerate(pr) if k not in usados and (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)) >= 0.5]
                    if m: usados.add(m[0]); ok += 1
                npred += len(pr); nver += len(ver)
    red.train()
    p = ok / max(npred, 1); r = ok / max(nver, 1)
    return L / len(va), p, r, 2 * p * r / max(p + r, 1e-9)


mejor = -1; paso = 0
for ep in range(EP):
    red.train(); perm = rng.permutation(len(tr)); acc = np.zeros(3); t1 = time.time()
    for i in range(0, len(perm), B):
        idx = np.sort(perm[i:i + B])
        x = aumentar(Xtr[idx]); obj = [torch.from_numpy(a[idx]) for a in (Ctr, Dtr, Mtr, Ytr, Wtr)]
        loss, partes = DCe.perdida(red(x), obj, pc)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(red.parameters(), 5.0); opt.step(); sch.step()
        acc += partes; paso += 1
    n = math.ceil(len(perm) / B)
    lv, p, r, f1 = evaluar()
    print(f'epoch {ep + 1}/{EP}: centre {acc[0] / n:.3f} dist {acc[1] / n:.3f} class {acc[2] / n:.3f} | val loss {lv:.3f} '
          f'P {p:.3f} R {r:.3f} F1 {f1:.3f} | {time.time() - t1:.0f} s', flush=True)
    if f1 > mejor:
        mejor = f1; torch.save(dict(estado=red.state_dict(), epoca=ep + 1, f1_val=f1, vecinas=DCe.VECINAS,
                                    ancho=D['ANCHO'], largo=LARGO), salida)
print('done; best validation F1', round(mejor, 3), f'{time.time() - t0:.0f} s')
