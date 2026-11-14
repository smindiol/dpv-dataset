"""Dataset B generation: high-efficiency domain per absorber material (network B: design + material inputs).

Origin (screening sweep 2026-11-14, salidas/barrido150, 900 devices over 6 materials on discrete levels):
- Perovskite (deltapv psc.py, mu = 2) and GaAs (mp = 10) are dropped: short diffusion length, median kappa_0 of
  5.1 and 1.3, median PCE of 1.8 % and 6.2 %, and 20-59 % Newton failures.
- kappa_0 = int R / int (G - R) at V = 0 is ordered by L_n / L_D (Spearman 0.90; PCE -0.86). Every device of the top
  10 % in PCE had L_n / L_D <= 0.54 and kappa_0 <= 0.13. Inside the box below, 100 % of the sweep's devices had
  kappa_0 <= 0.15, so kappa_0 is not computed here: the box enforces it by design and the data are filtered after.
- Box from the levels where devices met PCE >= 17 % and kappa_0 <= 0.15: tau >= 1e-7 s, mobility factor >= 0.3,
  N_d <= 3e17; L_n bounded by the physical criterion L_n / L_D <= 0.5.
- Eg is a property of each real material and stays at its value (InP 1.344, CdTe 1.5, Si 1.12 eV); only the generic
  E1 material, which stands for no specific compound, sweeps Eg in [1.3, 1.7] eV (dEg in [-0.2, 0.2]).

Sampling: a scrambled Sobol sequence, uniform on each range (log10-uniform for tau, mobility factor and dopings),
points outside L_n / L_D <= 0.5 rejected, so the sample is uniform over the feasible region. N devices per material;
a device whose simulation fails is replaced by the next feasible Sobol point (the reserve), at most MAX_EXTRA
replacements, so failures do not leave a hole and the set stays a Sobol prefix. Every device, failed ones included,
is stored with its state.
"""
import json
import math
import os
import sys
import time

import numpy as np

from . import barrido

LNLD_MAX = 0.5
N_DEF, MAX_EXTRA = 500, 100

COMUNES = [  # name, scale, lo, hi
    ("tau", "log", 1e-7, 1e-6),
    ("f_mu", "log", 0.3, 1.0),
    ("Nd_n", "log", 1e15, 3e17),
    ("Na_p", "log", 1e15, 1e18),
    ("L_n", "lin", 0.25e-4, 3e-4),
    ("L_p", "lin", 0.5e-4, 2e-4),
]
CAJAS = {
    "InP": COMUNES,
    "CdTe": COMUNES,
    "Si": COMUNES,
    "E1gen": [("dEg", "lin", -0.2, 0.2)] + COMUNES,
}
SEMILLA = {"InP": 3201, "E1gen": 3202, "CdTe": 3203, "Si": 3204}


def _valor(esc, lo, hi, u):
    return 10 ** (math.log10(lo) + u * (math.log10(hi) - math.log10(lo))) if esc == "log" else lo + u * (hi - lo)


def plan(material, n):
    """The first n Sobol points of `material` inside its box with L_n / L_D <= 0.5 (dEg = 0 unless swept)."""
    from scipy.stats import qmc
    caja = CAJAS[material]
    u = qmc.Sobol(d=len(caja), scramble=True, seed=SEMILLA[material]).random_base2(m=16)
    tareas = []
    for i, ui in enumerate(u):
        xi = {"material": material, "dEg": 0.0, **{c[0]: _valor(c[1], c[2], c[3], float(v)) for c, v in zip(caja, ui)}}
        if barrido.derivados(xi)["Ln_LD"] > LNLD_MAX:
            continue
        tareas.append({"idx": i, "tipo": "sobol", "familia": "BAR", "etapa": material, "u": ui.tolist(),
                       "factor_malla": 1, "xi": xi})
        if len(tareas) == n:
            return tareas
    raise RuntimeError(f"{material}: only {len(tareas)} feasible points in 2^16")


ESCALARES = ["eff", "jsc", "voc", "ff", "pmax", "vmax", "n_voltajes", "seg_total"]
ARREGLOS = ["V", "J", "G", "phi_eq", "pots", "Jn", "Jp", "newton_convergio"]


def _guardar(f, r, xi, reemplazo):
    g = f.create_group(f"dev_{r['idx']:06d}")
    g.attrs["estado"], g.attrs["mensaje"] = r["estado"], str(r.get("mensaje", ""))[:2000]
    g.attrs["reemplazo"], g.attrs["xi"] = reemplazo, json.dumps(xi)
    for k, v in barrido.derivados(xi).items():
        g.attrs[k] = v
    for k in ESCALARES:
        if k in r:
            g.attrs[k] = float(r[k])
    for k in ARREGLOS:
        if k in r:
            g.create_dataset(k, data=np.asarray(r[k]), compression="gzip", compression_opts=4)


def _log(msg):
    print(msg, flush=True)
    sys.stderr.flush()


def run(material, n=N_DEF, max_extra=MAX_EXTRA, salida=".", cada=50, trabajadores=None):
    """Simulate n devices of `material` on every CPU core into dpvB_<material>.h5; failed devices are replaced by
    reserve Sobol points (at most max_extra). Progress every `cada` finished devices and on every replacement."""
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    import h5py

    from . import simulacion
    todas = plan(material, n + max_extra)
    cola, reserva = todas[:n], todas[n:]
    trabajadores = trabajadores or os.cpu_count()
    ruta = os.path.join(salida, f"dpvB_{material}.h5")
    t0 = time.time()
    hechos = ok = fallos = extra = 0
    _log(f"[gen-B {material}] target {n} ok devices · reserve {len(reserva)} · {trabajadores} processes -> {ruta}")
    with h5py.File(ruta, "w") as f:
        f.attrs.update({"material": material, "caja": json.dumps(CAJAS[material]),
                        "material_fijo": json.dumps(barrido.MATERIALES[material]),
                        "criterio": json.dumps({"Ln_LD_max": LNLD_MAX, "max_extra": max_extra}),
                        "semilla_sobol": SEMILLA[material], "deltapv_commit": "f874e24"})
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(trabajadores, mp_context=ctx, initializer=simulacion.init_trabajador,
                                 initargs=("cpu", 0, None, 1)) as pool:
            fut = {pool.submit(simulacion.simular, t): (t, False) for t in cola}
            pend = set(fut)
            while pend:
                listos, pend = wait(pend, return_when=FIRST_COMPLETED)
                for x in listos:
                    t, es_extra = fut.pop(x)
                    try:
                        r = x.result()
                    except Exception as e:  # noqa: BLE001  (a dead worker must not stop the run)
                        r = {"idx": t["idx"], "estado": "error", "mensaje": repr(e)}
                    _guardar(f, r, t["xi"], es_extra)
                    hechos += 1
                    if r["estado"] == "ok":
                        ok += 1
                    else:
                        fallos += 1
                        if reserva:
                            nt = reserva.pop(0)
                            extra += 1
                            nf = pool.submit(simulacion.simular, nt)
                            fut[nf] = (nt, True)
                            pend.add(nf)
                            _log(f"[gen-B {material}] device {t['idx']} failed ({r['estado']}) -> replacement "
                                 f"{extra}/{max_extra} (Sobol {nt['idx']})")
                        else:
                            _log(f"[gen-B {material}] device {t['idx']} failed ({r['estado']}); reserve exhausted")
                    if hechos % cada == 0 or not pend:
                        f.flush()
                        dt = time.time() - t0
                        resta = len(pend)
                        _log(f"[gen-B {material}] {ok}/{n} ok · {resta} left · failed {fallos} · replacements "
                             f"{extra}/{max_extra} · {dt / hechos:.1f} s/device · ETA {dt / hechos * resta / 60:.0f} min")
        res = {"material": material, "target": n, "ok": ok, "failed": fallos, "replacements": extra,
               "hours": (time.time() - t0) / 3600}
        f.attrs["resumen"] = json.dumps(res)
    _log(f"[gen-B {material}] DONE {json.dumps(res)}")
    return res
