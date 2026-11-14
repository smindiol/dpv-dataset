"""Dataset B generation: high-efficiency, low-kappa_0 domain per absorber material (network B: design + material inputs).

Origin (screening sweep 2026-11-14, salidas/barrido150, 900 devices over 6 materials on discrete levels):
- Perovskite (deltapv psc.py, mu = 2) and GaAs (mp = 10) are dropped: short diffusion length, median kappa_0 of
  5.1 and 1.3, median PCE of 1.8 % and 6.2 %, and 20-59 % Newton failures.
- kappa_0 = int R / int (G - R) at V = 0 is ordered by L_n / L_D (Spearman 0.90; PCE -0.86). Every device of the top
  10 % in PCE had L_n / L_D <= 0.54 and kappa_0 <= 0.13.
- Per material, the levels where devices met PCE >= 17 % and kappa_0 <= 0.15 set the box below: tau >= 1e-7 s
  (shorter tau: 0-32 % pass), mobility factor >= 0.3 (0.1: 6-33 %), N_d <= 3e17 (1e18 raised kappa_0 to 3), Si with
  dEg >= 0 (dEg < 0: 0-17 %). L_n is bounded by the physical criterion L_n / L_D <= 0.5, not by observed failures.

Sampling: a scrambled Sobol sequence, uniform on each range (log10-uniform for tau, mobility factor and dopings),
with points outside L_n / L_D <= 0.5 rejected, so the sample is uniform over the feasible region. Nothing is filtered
by outcome: every device is stored with its state, PCE and kappa_0, plus the flag `cumple` (PCE >= 17 % and
kappa_0 <= 0.15). N per material is sized from the pass rate measured in the sweep inside this same box so that at
least 1000 devices meet `cumple`.
"""
import json
import math
import os
import time

import numpy as np

from . import barrido

Q = 1.602176634e-19
EFF_MIN, KAPPA_MAX, LNLD_MAX = 0.17, 0.15, 0.5

COMUNES = [  # name, scale, lo, hi
    ("tau", "log", 1e-7, 1e-6),
    ("f_mu", "log", 0.3, 1.0),
    ("Nd_n", "log", 1e15, 3e17),
    ("Na_p", "log", 1e15, 1e18),
    ("L_n", "lin", 0.25e-4, 3e-4),
    ("L_p", "lin", 0.5e-4, 2e-4),
]
CAJAS = {
    "InP": [("dEg", "lin", -0.2, 0.2)] + COMUNES,
    "E1gen": [("dEg", "lin", -0.2, 0.2)] + COMUNES,
    "CdTe": [("dEg", "lin", -0.2, 0.2)] + COMUNES,
    "Si": [("dEg", "lin", 0.0, 0.2)] + COMUNES,
}
# measured in the sweep inside the box: pass = convergence x (PCE >= 17 % and kappa_0 <= 0.15)
PASE_BARRIDO = {"InP": 1.00, "E1gen": 1.00, "CdTe": 0.71, "Si": 0.61 * 0.76}
N = {"InP": 1200, "E1gen": 1200, "CdTe": 1600, "Si": 2400}
SEMILLA = {"InP": 3101, "E1gen": 3102, "CdTe": 3103, "Si": 3104}


def _valor(esc, lo, hi, u):
    return 10 ** (math.log10(lo) + u * (math.log10(hi) - math.log10(lo))) if esc == "log" else lo + u * (hi - lo)


def plan(material, n=None):
    """The first n Sobol points of `material` inside the box that satisfy L_n / L_D <= 0.5."""
    from scipy.stats import qmc
    n = N[material] if n is None else n
    caja = CAJAS[material]
    u = qmc.Sobol(d=len(caja), scramble=True, seed=SEMILLA[material]).random_base2(m=16)
    tareas = []
    for i, ui in enumerate(u):
        xi = {"material": material, **{c[0]: _valor(c[1], c[2], c[3], float(v)) for c, v in zip(caja, ui)}}
        if barrido.derivados(xi)["Ln_LD"] > LNLD_MAX:
            continue
        tareas.append({"idx": i, "tipo": "sobol", "familia": "BAR", "etapa": material, "u": ui.tolist(),
                       "factor_malla": 1, "xi": xi})
        if len(tareas) == n:
            return tareas
    raise RuntimeError(f"{material}: only {len(tareas)} feasible points in 2^16")


def simular_b(tarea):
    """deltapv simulation plus J_gen = q int G dx and kappa_0 = (J_gen - J_sc) / J_sc."""
    from . import simulacion
    r = simulacion.simular(tarea)
    try:
        dpv = simulacion.DPV
        from deltapv import optical, scales
        des, ls = barrido.construir_barrido(dpv, tarea["xi"])
        G = np.asarray(optical.compute_G(des, ls)) * float(scales.gratedens)
        r["J_gen"] = Q * float(np.trapezoid(G, np.asarray(des.grid) * float(scales.length))) * 1e4   # A/m^2
        r["x_cm"] = np.asarray(des.grid) * float(scales.length)
    except Exception as e:  # noqa: BLE001
        r["J_gen"] = float("nan")
        r["mensaje"] += f" | J_gen: {e}"
    ok = r["estado"] == "ok" and r.get("jsc", 0) > 0
    r["kappa0"] = (r["J_gen"] - r["jsc"]) / r["jsc"] if ok else float("nan")
    r["cumple"] = bool(ok and r["eff"] >= EFF_MIN and r["kappa0"] <= KAPPA_MAX)
    return r


ESCALARES = ["eff", "jsc", "voc", "ff", "pmax", "vmax", "J_gen", "kappa0", "n_voltajes", "seg_total"]
ARREGLOS = ["V", "J", "x_cm", "G", "phi_eq", "pots", "Jn", "Jp", "newton_convergio"]


def _guardar(f, r, xi):
    g = f.create_group(f"dev_{r['idx']:06d}")
    g.attrs["estado"], g.attrs["mensaje"], g.attrs["cumple"] = r["estado"], r["mensaje"][:2000], r["cumple"]
    g.attrs["xi"] = json.dumps(xi)
    for k, v in barrido.derivados(xi).items():
        g.attrs[k] = v
    for k in ESCALARES:
        if k in r:
            g.attrs[k] = float(r[k])
    for k in ARREGLOS:
        if k in r:
            g.create_dataset(k, data=np.asarray(r[k]), compression="gzip", compression_opts=4)


def run(material, n=None, salida=".", cada=50, trabajadores=None):
    """Simulate `material`'s plan on every CPU core, one HDF5 file, progress every `cada` devices."""
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    import h5py

    from . import simulacion
    tareas = plan(material, n)
    trabajadores = trabajadores or os.cpu_count()
    ruta = os.path.join(salida, f"dpvB_{material}.h5")
    t0, hechos, n_ok, n_cumple = time.time(), 0, 0, 0
    print(f"[gen-B {material}] {len(tareas)} devices · {trabajadores} processes -> {ruta}", flush=True)
    with h5py.File(ruta, "w") as f:
        f.attrs.update({"material": material, "caja": json.dumps(CAJAS[material]),
                        "material_fijo": json.dumps({k: v for k, v in barrido.MATERIALES[material].items()}),
                        "criterio": json.dumps({"eff_min": EFF_MIN, "kappa0_max": KAPPA_MAX, "Ln_LD_max": LNLD_MAX}),
                        "semilla_sobol": SEMILLA[material], "deltapv_commit": "f874e24"})
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(trabajadores, mp_context=ctx, initializer=simulacion.init_trabajador,
                                 initargs=("cpu", 0, None, 1)) as pool:
            fut = {pool.submit(simular_b, t): t for t in tareas}
            pend = set(fut)
            while pend:
                listos, pend = wait(pend, return_when=FIRST_COMPLETED)
                for x in listos:
                    t = fut[x]
                    try:
                        r = x.result()
                    except Exception as e:  # noqa: BLE001  (a dead worker must not stop the run)
                        r = {"idx": t["idx"], "estado": "error", "mensaje": repr(e), "cumple": False}
                    _guardar(f, r, t["xi"])
                    hechos += 1
                    n_ok += r["estado"] == "ok"
                    n_cumple += bool(r["cumple"])
                    if hechos % cada == 0 or hechos == len(tareas):
                        f.flush()
                        dt = time.time() - t0
                        print(f"[gen-B {material}] {hechos}/{len(tareas)} done · {len(tareas) - hechos} left · "
                              f"ok {n_ok} · meet PCE>=17% & kappa0<=0.15: {n_cumple} · "
                              f"{dt / hechos:.2f} s/device · ETA {dt / hechos * (len(tareas) - hechos) / 60:.0f} min",
                              flush=True)
        res = {"material": material, "n": hechos, "ok": n_ok, "cumple": n_cumple, "hours": (time.time() - t0) / 3600}
        f.attrs["resumen"] = json.dumps(res)
    print(f"[gen-B {material}] DONE {json.dumps(res)}", flush=True)
    return res
