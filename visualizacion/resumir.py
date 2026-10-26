"""Resume los .h5 de dpv-dataset/salidas en un JSON compacto para la visualizacion.

Uso: python -I resumir.py <dir_salidas> <json_salida>
Un bloque repetido conserva la malla mas fina y, a igual malla, el completo.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import h5py
import numpy as np

raiz, destino = sys.argv[1], sys.argv[2]

# un archivo por (familia, etapa, bloque): gana la malla base mas fina y, con la misma
# malla, el marcado completo (E2/E4 x4 en m4-s*/ reemplazan a los x1 de s3/s4)
elegidos = {}
for ruta in sorted(glob.glob(os.path.join(raiz, "*", "dpv_*.h5"))):
    with h5py.File(ruta, "r") as f:
        clave = (f.attrs["familia"], f.attrs["etapa"], int(f.attrs["bloque"]))
        orden = (int(f.attrs.get("factor_malla_base", 1)), bool(f.attrs["completo"]))
    if clave not in elegidos or orden > elegidos[clave][2]:
        elegidos[clave] = (ruta, orden[1], orden)

rng = np.random.default_rng(0)
fe_info = {}
puntos = defaultdict(lambda: defaultdict(list))
curvas = defaultdict(list)
perfiles = defaultdict(list)
piso = defaultdict(list)
bloques = []

for (fam, et, b), (ruta, completo, _orden) in sorted(elegidos.items()):
    fe = fam + et
    with h5py.File(ruta, "r") as f:
        if fe not in fe_info:
            fe_info[fe] = {
                "familia": fam, "etapa": et,
                "descripcion": str(f.attrs["descripcion"]),
                "parametros": [p["nombre"] for p in json.loads(f.attrs["parametros_xi"])],
                "rangos": json.loads(f.attrs["parametros_xi"]),
                "estados": Counter(), "esquinas": Counter(), "refinados": 0,
                "mb": 0.0, "bloques": [],
            }
        info = fe_info[fe]
        info["mb"] += os.path.getsize(ruta) / 1e6
        info["bloques"].append(b)
        est_bloque = Counter()
        base_eff = {}
        for k in f["sims"]:
            g = f["sims"][k]
            a = g.attrs
            estado = str(a["estado"])
            info["estados"][estado] += 1
            est_bloque[estado] += 1
            p = puntos[fe]
            p["u"].append([round(float(x), 4) for x in g["u"][()]])
            p["estado"].append(estado)
            ok = estado == "ok"
            p["eff"].append(round(100 * float(a["eff"]), 4) if ok else None)
            p["voc"].append(round(float(a["voc"]), 4) if ok else None)
            p["jsc"].append(round(float(a["jsc"]) / 10, 3) if ok else None)  # mA/cm^2
            p["ff"].append(round(100 * float(a["ff"]), 3) if ok else None)
            p["seg"].append(round(float(a.get("seg_total", 0.0)), 1))
            if ok:
                base_eff[k] = float(a["eff"])
                if len(curvas[fe]) < 14 and rng.random() < 0.02:
                    curvas[fe].append({
                        "V": np.round(g["V"][()], 4).tolist(),
                        "J": np.round(-10 * g["J"][()] * 100, 4).tolist(),  # mA/cm^2, signo fotocorriente +
                        "eff": round(100 * float(a["eff"]), 2),
                    })
                if len(perfiles[fe]) < 6 and "phi_eq" in g and rng.random() < 0.01:
                    x = g["x_cm"][()] * 1e4  # um
                    perfiles[fe].append({
                        "x": np.round(x, 5).tolist(),
                        "phi": np.round(g["phi_eq"][()], 4).tolist(),
                        "eff": round(100 * float(a["eff"]), 2),
                    })
        for k in f["esquinas"]:
            info["esquinas"][str(f["esquinas"][k].attrs["estado"])] += 1
        for k in f["refinados"]:
            a = f["refinados"][k].attrs
            info["refinados"] += 1
            if str(a["estado"]) == "ok" and k in base_eff:
                piso[fe].append(abs(100 * float(a["eff"]) - 100 * base_eff[k]))
        bloques.append({"fe": fe, "bloque": b, "completo": completo,
                        "archivo": os.path.relpath(ruta, raiz), "estados": dict(est_bloque)})

salida = {"familias": {}, "bloques": bloques}
for fe, info in sorted(fe_info.items()):
    d = piso.get(fe, [])
    salida["familias"][fe] = {
        **{k: v for k, v in info.items() if k not in ("estados", "esquinas")},
        "estados": dict(info["estados"]), "esquinas": dict(info["esquinas"]),
        "mb": round(info["mb"], 1),
        "piso_q95_pp": round(float(np.quantile(d, 0.95)), 4) if d else None,
        "piso_n": len(d),
        "puntos": puntos[fe], "curvas": curvas[fe], "perfiles": perfiles[fe],
    }
with open(destino, "w") as fh:
    json.dump(salida, fh, separators=(",", ":"))
for fe, v in salida["familias"].items():
    print(fe, v["estados"], "esquinas", v["esquinas"], "refinados", v["refinados"],
          "piso_q95", v["piso_q95_pp"], f"{v['mb']:.0f} MB", "bloques", sorted(v["bloques"]))
print(os.path.getsize(destino) / 1e6, "MB json")
