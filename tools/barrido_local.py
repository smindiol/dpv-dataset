"""Run the discrete screening sweep (dpvgen/barrido.py) on the local CPU, one process per core.

Each process runs JAX on CPU with one thread (the solver works on 500-node systems, too small for
intra-op threading to pay off; one simulation per core does). The XLA compilations are cached on disk
(JAX_COMPILATION_CACHE_DIR), so a restart does not compile again. Results go to one JSON line per device,
appended as soon as each device finishes: interrupt at any time and run the same command to resume.

  python tools/barrido_local.py --prueba 2              # short check: 2 devices per material, throughput
  python tools/barrido_local.py                         # full sweep: 512 per material
  python tools/barrido_local.py --materiales GaAs Si --trabajadores 8
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from dpvgen import barrido, simulacion  # noqa: E402

ESCALARES = ("estado", "mensaje", "eff", "jsc", "voc", "ff", "pmax", "n_voltajes", "seg_total")


def fila(r, xi):
    out = {"idx": r["idx"], "material": xi["material"], **{k: xi[k] for k in barrido.EJES},
           **barrido.derivados(xi)}
    for k in ESCALARES:
        v = r.get(k)
        out[k] = v if isinstance(v, str) or v is None else float(v)
    if "V" in r:
        out["V"], out["J"] = r["V"].tolist(), r["J"].tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materiales", nargs="*", default=list(barrido.MATERIALES))
    ap.add_argument("--n", type=int, default=barrido.N_POR_MATERIAL)
    ap.add_argument("--prueba", type=int, default=0, help="devices per material for a short check")
    ap.add_argument("--trabajadores", type=int, default=os.cpu_count())
    ap.add_argument("--salida", default=str(RAIZ / "salidas" / "barrido"))
    a = ap.parse_args()

    n = a.prueba or a.n
    sal = Path(a.salida) / ("prueba" if a.prueba else "completo")
    sal.mkdir(parents=True, exist_ok=True)
    cache = RAIZ / "salidas" / "jax-cache"
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

    tareas = []
    for m in a.materiales:
        p = sal / f"{m}.jsonl"
        hechos = {json.loads(l)["idx"] for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()
        tareas += [t for t in barrido.plan(m, n) if t["idx"] not in hechos]
    print(f"[barrido] {len(tareas)} pending devices, {a.trabajadores} processes -> {sal}", flush=True)
    if not tareas:
        return

    t0, hechos, estados = time.time(), 0, {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(a.trabajadores, mp_context=ctx, initializer=simulacion.init_trabajador,
                             initargs=("cpu", 0, None, 1)) as pool:
        fut = {pool.submit(simulacion.simular, t): t for t in tareas}
        pend = set(fut)
        while pend:
            listos, pend = wait(pend, return_when=FIRST_COMPLETED)
            for f in listos:
                t = fut[f]
                r = f.result()
                with open(sal / f"{t['xi']['material']}.jsonl", "a") as fh:
                    fh.write(json.dumps(fila(r, t["xi"])) + "\n")
                hechos += 1
                estados[r["estado"]] = estados.get(r["estado"], 0) + 1
                dt = time.time() - t0
                eta = dt / hechos * (len(tareas) - hechos)
                print(f"[barrido] {hechos}/{len(tareas)} {t['xi']['material']} eff={r.get('eff', float('nan')):.4f} "
                      f"{r['estado']} | {dt / hechos:.2f} s/device | ETA {eta / 60:.1f} min", flush=True)
    dt = time.time() - t0
    resumen = {"devices": hechos, "seconds": dt, "s_per_device": dt / hechos, "processes": a.trabajadores,
               "states": estados, "full_sweep_hours_estimate":
                   dt / hechos * barrido.N_POR_MATERIAL * len(barrido.MATERIALES) / 3600}
    (sal / "resumen.json").write_text(json.dumps(resumen, indent=2))
    print(f"[barrido] {json.dumps(resumen)}", flush=True)


if __name__ == "__main__":
    main()
