"""Convierte dpv_resumen.json en las tablas que lee el tablero.

Uso: python -I tablas.py <dpv_resumen.json> <dir_salida>
"""
import csv
import json
import math
import os
import sys

import numpy as np

resumen = json.load(open(sys.argv[1]))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)


def valor_real(p, u):
    if p["escala"] == "log":
        a, b = math.log10(p["min"]), math.log10(p["max"])
        return 10 ** (a + u * (b - a))
    return p["min"] + u * (p["max"] - p["min"])


familias, dispositivos, curvas, perfiles, parametros, bloques = [], [], [], [], [], []
for fe, v in resumen["familias"].items():
    pts = v["puntos"]
    pce = np.array([x for x in pts["eff"] if x is not None])
    ok = v["estados"].get("ok", 0)
    total = sum(v["estados"].values())
    def med(k):
        a = [x for x in pts[k] if x is not None]
        return round(float(np.median(a)), 4) if a else None
    familias.append({
        "fe": fe, "familia": v["familia"], "etapa": v["etapa"], "descripcion": v["descripcion"],
        "d": len(v["parametros"]), "simulados": total, "ok": ok, "fallos": total - ok,
        "tasa_fallo": round((total - ok) / total, 4),
        "esquinas_ok": v["esquinas"].get("ok", 0),
        "esquinas_fallo": sum(c for e, c in v["esquinas"].items() if e != "ok"),
        "refinados": v["refinados"], "piso_q95_pp": v["piso_q95_pp"],
        "mb": v["mb"], "bloques": len(v["bloques"]),
        "pce_p5": round(float(np.percentile(pce, 5)), 3),
        "pce_p50": round(float(np.percentile(pce, 50)), 3),
        "pce_p95": round(float(np.percentile(pce, 95)), 3),
        "voc_p50": med("voc"), "jsc_p50": med("jsc"), "ff_p50": med("ff"),
    })
    for i, p in enumerate(v["rangos"]):
        parametros.append({"fe": fe, "orden": i + 1, "nombre": p["nombre"], "escala": p["escala"],
                           "min": p["min"], "max": p["max"], "nominal": p["nominal"],
                           "etapa_param": p["etapa"], "fuente": p["fuente"]})
    for j in range(len(pts["estado"])):
        fila = {"fe": fe, "estado": pts["estado"][j], "pce": pts["eff"][j], "voc": pts["voc"][j],
                "jsc": pts["jsc"][j], "ff": pts["ff"][j]}
        for i in range(5):
            if i < len(v["rangos"]):
                fila[f"p{i + 1}"] = float(f"{valor_real(v['rangos'][i], pts['u'][j][i]):.5g}")
            else:
                fila[f"p{i + 1}"] = ""
        dispositivos.append(fila)
    for c, cur in enumerate(v["curvas"]):
        J = np.array(cur["J"])
        signo = 1.0 if J[0] > 0 else -1.0  # fotocorriente positiva en V = 0
        for V, Jv in zip(cur["V"], J):
            curvas.append({"fe": fe, "curva": c + 1, "V": V, "J": round(signo * Jv, 4), "pce": cur["eff"]})
    for c, per in enumerate(v["perfiles"]):
        for x, phi in zip(per["x"], per["phi"]):
            perfiles.append({"fe": fe, "perfil": c + 1, "x_um": x, "phi": phi, "pce": per["eff"]})

for b in resumen["bloques"]:
    bloques.append({"fe": b["fe"], "bloque": f"b{b['bloque']:03d}",
                    "sesion": b["archivo"].split("/")[0], "completo": b["completo"],
                    "ok": b["estados"].get("ok", 0),
                    "fallos": sum(c for e, c in b["estados"].items() if e != "ok")})


def escribir(nombre, filas):
    with open(os.path.join(out, nombre), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(nombre, len(filas), "filas", round(os.path.getsize(os.path.join(out, nombre)) / 1e6, 2), "MB")


json.dump(familias, open(os.path.join(out, "familias.json"), "w"), ensure_ascii=False, indent=0)
print("familias.json", len(familias))
escribir("dispositivos.csv", dispositivos)
escribir("curvas_jv.csv", curvas)
escribir("perfiles_phi.csv", perfiles)
escribir("parametros.csv", parametros)
escribir("bloques.csv", bloques)
