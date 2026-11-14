"""Descriptive analysis of the screening sweep: efficiency, kappa_0 and convergence vs every swept variable.

kappa_0 = int R / int (G - R) at V = 0, from the stored J_sc and the generation current
J_gen = q int G dx (deltapv's own G for that design): kappa_0 = (J_gen - J_sc) / J_sc (contact terms neglected).
Writes salidas/barrido150/analisis.json (one row per device, profiles excluded).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
Q = 1.602176634e-19


def jgen(xi):
    import contextlib, io, os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        import deltapv as dpv
        from deltapv import optical, scales
        from dpvgen import barrido
    des, ls = barrido.construir_barrido(dpv, xi)
    G = np.asarray(optical.compute_G(des, ls)) * float(scales.gratedens)   # cm^-3 s^-1
    return Q * float(np.trapezoid(G, np.asarray(des.grid) * float(scales.length))) * 1e4             # A/m^2


def main(carpeta):
    filas = []
    for p in sorted(Path(carpeta).glob("*.jsonl")):
        for l in p.read_text().splitlines():
            r = json.loads(l)
            r.pop("V", None), r.pop("J", None)
            filas.append(r)
    xis = [{k: r[k] for k in ("material", "dEg", "f_mu", "tau", "Nd_n", "Na_p", "L_n", "L_p")} for r in filas]
    with ProcessPoolExecutor(12) as ex:
        jg = list(ex.map(jgen, xis, chunksize=8))
    for r, j in zip(filas, jg):
        r["J_gen"] = j
        ok = r["estado"] == "ok" and r["jsc"] and r["jsc"] > 0
        r["kappa0"] = (j - r["jsc"]) / r["jsc"] if ok else None
    out = Path(carpeta).parent / "analisis.json"
    out.write_text(json.dumps(filas))
    print(out, len(filas))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(RAIZ / "salidas/barrido150/completo"))
