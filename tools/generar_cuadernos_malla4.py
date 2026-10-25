"""Genera los cuadernos de re-simulacion de E2/E4 con la malla base x4 (720 nodos).

Uso: python tools/generar_cuadernos_malla4.py <commit>
"""
import json, sys

COMMIT = sys.argv[1]
SESIONES = {
    "s5": ["E2A-b000", "E2A-b001"], "s6": ["E2B-b000", "E2B-b001"],
    "s7": ["E4A-b000", "E4A-b001"], "s8": ["E4B-b000", "E4B-b001"],
    "s9": ["E2A-b002", "E2B-b002"], "s10": ["E4A-b002", "E4B-b002"],
    "todo": [f"{fe}-b{b:03d}" for fe in ("E2A", "E2B", "E4A", "E4B") for b in range(3)],
}


def celda(tipo, texto):
    c = {"cell_type": tipo, "metadata": {}, "source": texto.strip("\n").splitlines(True)}
    if tipo == "code":
        c.update(outputs=[], execution_count=None)
    return c


for nombre, unidades in SESIONES.items():
    md = f"""
# Re-simulacion de E2/E4 con malla base x4 — {nombre}

Sustituye los bloques de E2 y E4 hechos con la malla graduada de ex2 (180 nodos), cuyo piso de
discretizacion (Ec. 21) era de 0.20 a 0.48 pp, por la misma malla con 4 veces los nodos (720). La
prueba de convergencia del 2026-10-25 dio, en los 16 dispositivos de peor piso, |PCE(x4) - PCE(x6)|
de 0.055 pp como maximo, y las 60 esquinas de E2/E4 resolvieron con x4. El refinado de la Ec. 21
(5 % del bloque b000) se hace con x16.

Bloques: {", ".join(f"`{u}`" for u in unidades)}. Salida: `dpv_<fe>_m4_b<nnn>.h5` por bloque y `resumen_sesion.json`.

**Donde correrlo** (solo CPU, la GPU no se usa):
- **Kaggle**: Accelerator = None, Internet = On, *Save Version → Save & Run All*. Cada sesion `s5`..`s10` cabe en las 11 h.
- **Colab**: entorno de solo CPU; la salida se copia a Google Drive (`MyDrive/dpv-dataset/salidas_m4`) al terminar cada bloque, y al volver a abrir el cuaderno se retoma donde quedo.
- **Local**: corre en el directorio actual con un proceso por nucleo.
"""
    c1 = f'''
import os, sys, subprocess, pathlib
COMMIT = "{COMMIT}"
UNIDADES = {json.dumps(unidades)}
EN_KAGGLE = os.path.isdir("/kaggle/working")
EN_COLAB = "google.colab" in sys.modules or os.path.isdir("/content")
ROOT = pathlib.Path("/tmp/dpv-dataset")
if not ROOT.exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/smindiol/dpv-dataset", str(ROOT)], check=True)
subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin"], check=True)
subprocess.run(["git", "-C", str(ROOT), "checkout", "-q", COMMIT], check=True)
sys.path.insert(0, str(ROOT))
os.environ["FORGE_CLONE_COMMIT"] = COMMIT
print(subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
'''
    c2 = '''
# deltapv fijado al commit verificado; --no-deps para no tocar el jax de la imagen
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                "git+https://github.com/romanodev/deltapv@f874e24"], check=True)
import h5py, numpy, jax
print("jax", jax.__version__, "| numpy", numpy.__version__, "| h5py", h5py.__version__, "| CPUs", os.cpu_count())

DRIVE = None
if EN_KAGGLE:
    DIR_SALIDA = "/kaggle/working"
elif EN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = "/content/drive/MyDrive/dpv-dataset/salidas_m4"
    os.makedirs(DRIVE, exist_ok=True)
    DIR_SALIDA = "/content/salidas_m4"
    os.makedirs(DIR_SALIDA, exist_ok=True)
    subprocess.run(f'cp -n "{DRIVE}"/*.h5 "{DIR_SALIDA}"/ 2>/dev/null', shell=True)   # retomar
else:
    DIR_SALIDA = os.path.join(os.getcwd(), "salidas_m4")
    os.makedirs(DIR_SALIDA, exist_ok=True)
print("salida en", DIR_SALIDA, "| copia en Drive:", DRIVE)
'''
    c3 = '''
# Comprobacion antes de producir: la esquina que fallaba por el nodo sin material y un
# dispositivo cuyo valor x4 ya existe (refinado de E2A-b000, idx 9, Kaggle s3).
from dpvgen import orquestador, simulacion as s
s.init_trabajador("cpu", 0, None, 1)
u = [0.0, 0.0, 1.0, 0.0]
r = s.simular({"idx": -5, "tipo": "esquina", "familia": "E2", "etapa": "B", "u": u,
               "xi": s.xi_desde_unitario("E2", "B", u), "factor_malla": 4})
print("esquina E2B -5 x4:", r["estado"], round(100 * r.get("eff", float("nan")), 4), "% (Kaggle x1: 7.8059 %)")
t = [t for t in s.plan_de_bloque("E2", "A", 0, 0.05, s.TAM_BLOQUE, 1) if t["tipo"] == "refinado" and t["idx"] == 9][0]
r2 = s.simular(t)
print("E2A idx 9 x4:", r2["estado"], repr(r2.get("eff")), "| Kaggle x4: 0.13349030110887902")
assert r["estado"] == "ok" and r2["estado"] == "ok" and abs(r2["eff"] - 0.13349030110887902) < 1e-4, "la comprobacion fallo: no producir"
'''
    c4 = '''
import shutil, glob
orquestador.configurar(PLATAFORMA="cpu", N_TRABAJADORES=None, PRESUPUESTO_HORAS=11.0 if EN_KAGGLE else 1000.0,
                       DIR_SALIDA=DIR_SALIDA, REFINAR_SOLO_BLOQUE0=True,
                       FACTOR_MALLA_BASE={"E2": 4, "E4": 4})
resumenes = []
for u in UNIDADES:
    resumenes.append(orquestador.ejecutar_unidades([u]))
    if DRIVE:
        for ruta in glob.glob(os.path.join(DIR_SALIDA, "*.h5")):
            shutil.copy2(ruta, DRIVE)
        print("copiado a Drive:", u, flush=True)
'''
    c5 = '''
# Verificacion de lo producido: estados (r04 exige reportarlos), PCE y piso de la Ec. 21
import json, numpy as np
from collections import Counter
for ruta in sorted(glob.glob(os.path.join(DIR_SALIDA, "dpv_*_m4_b*.h5"))):
    with h5py.File(ruta, "r") as f:
        print(f"\\n{os.path.basename(ruta)}  completo={f.attrs.get('completo')}  malla x{f.attrs.get('factor_malla_base')}  "
              f"{os.path.getsize(ruta)/1e9:.2f} GB")
        for g in ("sims", "esquinas", "refinados"):
            est = Counter(f[g][k].attrs["estado"] for k in f[g])
            print(f"  /{g}: {sum(est.values())}  {dict(est)}")
        oks = [k for k in f["sims"] if f["sims"][k].attrs["estado"] == "ok"]
        if oks:
            eff = np.array([f["sims"][k].attrs["eff"] for k in oks]) * 100
            print(f"  PCE %: p5={np.percentile(eff,5):.2f} p50={np.median(eff):.2f} p95={np.percentile(eff,95):.2f}")
        d = [abs(f["refinados"][k].attrs["eff"] - f["sims"][k].attrs["eff"]) * 100 for k in f["refinados"]
             if f["refinados"][k].attrs["estado"] == "ok" and k in f["sims"] and f["sims"][k].attrs["estado"] == "ok"]
        if d:
            print(f"  piso Ec. 21 (x4 frente a x16): Q95 = {np.quantile(d, 0.95):.4f} pp, max = {max(d):.4f} pp, n = {len(d)}")
'''
    nb = {"cells": [celda("markdown", md), celda("code", c1), celda("code", c2), celda("code", c3),
                    celda("code", c4), celda("code", c5)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    ruta = f"dataset/Notebooks/malla4_{nombre}.ipynb"
    json.dump(nb, open(ruta, "w"), indent=1, ensure_ascii=False)
    print(ruta, unidades)
