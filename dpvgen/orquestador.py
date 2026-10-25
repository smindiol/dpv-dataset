"""Orquestador del generador dPV: pool de procesos, escritura HDF5, modos TEST y PRODUCCION."""
import sys, os, json, math, time, shutil, subprocess, multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
import numpy as np
import h5py

from types import SimpleNamespace

from dpvgen import simulacion as dpvgen

# Configuracion de la corrida; el cuaderno la fija con configurar().
C = SimpleNamespace(
    FAMILIA="E1", ETAPA="A", N_TOTAL=12_500, FRACCION_REFINADO=0.05,
    # el refinado 4x tarda ~20 veces una simulacion normal; con True solo el bloque 0
    # de cada familia-etapa lo lleva, que basta para el piso de la Ec. 21
    REFINAR_SOLO_BLOQUE0=False,
    PLATAFORMA="cpu", N_TRABAJADORES=None, PRESUPUESTO_HORAS=11.0,
    TEST_FAMILIA="E2", TEST_ETAPA="B", N_TEST_SIMS=8,
    DELTAPV_COMMIT="f874e24", DIR_SALIDA=".",
)


def configurar(**kw):
    for k, v in kw.items():
        if not hasattr(C, k):
            raise KeyError(f"parametro desconocido: {k}")
        setattr(C, k, v)
    return C

CTX = mp.get_context("spawn")   # jax no tolera fork despues de inicializarse


def n_gpus_visibles():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return len([l for l in out.stdout.splitlines() if l.startswith("GPU")])
    except Exception:
        return 0


def crear_pool(plataforma, n_trabajadores):
    ng = n_gpus_visibles() if plataforma == "gpu" else 0
    if plataforma == "gpu" and ng == 0:
        raise RuntimeError("C.PLATAFORMA='gpu' pero no hay GPU visible (activa el acelerador del cuaderno)")
    contador = CTX.Value("i", 0)
    return ProcessPoolExecutor(max_workers=n_trabajadores, mp_context=CTX,
                               initializer=dpvgen.init_trabajador,
                               initargs=("cpu" if plataforma == "cpu" else "gpu", ng, contador, 1))


# -----------------------------------------------------------------------------
# Escritura HDF5: un archivo por familia-etapa-fragmento, un solo proceso escritor
# -----------------------------------------------------------------------------
ESCALARES = ["pmax", "vmax", "eff", "eff_dpv", "P_in_std", "P_in_luz", "jsc", "voc", "ff",
             "n_voltajes", "eq_niter", "eq_p", "eq_convergio", "seg_equilibrio", "seg_total",
             "factor_malla"]
ARREGLOS_SIEMPRE = ["u", "V", "J", "newton_niter", "newton_niter_aux", "newton_p", "newton_F",
                    "newton_denso", "newton_convergio", "seg_por_voltaje"]
PERFILES = ["x_cm", "G", "phi_eq", "pots", "Jn", "Jp"]


def abrir_h5(ruta, familia, etapa, n_total):
    nuevo = not os.path.exists(ruta)
    f = h5py.File(ruta, "a")
    if nuevo:
        f.attrs["familia"] = familia
        f.attrs["etapa"] = etapa
        f.attrs["descripcion"] = dpvgen.FAMILIAS[familia]["descripcion"]
        f.attrs["propuesta"] = "research-concept-...-r04.md, Seccion 7"
        f.attrs["deltapv_commit"] = C.DELTAPV_COMMIT
        f.attrs["semilla_sobol"] = dpvgen.SEMILLAS_SOBOL[familia + etapa]
        f.attrs["n_total_sobol"] = n_total
        f.attrs["parametros_xi"] = json.dumps(
            [{"nombre": p[0], "escala": p[1], "min": p[2], "max": p[3], "nominal": p[4],
              "etapa": p[5], "fuente": p[6]} for p in dpvgen.parametros_activos(familia, etapa)])
        f.attrs["parametros_fijos"] = json.dumps(
            {p[0]: p[4] for p in dpvgen.FAMILIAS[familia]["parametros"]
             if p not in dpvgen.parametros_activos(familia, etapa)})
        f.attrs["unidades"] = json.dumps({
            "xi": "dopajes cm^-3, espesores cm, eta_PL adimensional",
            "u": "xi normalizado a [0,1] por (Ec. 3): log10 en dopajes, lineal en espesores y eta_PL",
            "x_cm": "cm", "V": "V", "J": "A/cm^2 (iv de dPV)", "jsc": "A/m^2", "pmax": "W/m^2",
            "eff": "fraccion (Pmax / suma P_in del espectro estandar, Ec. 14)",
            "pots": "(n_voltajes, 3, N): phi, phi_n, phi_p adimensionales (unidades de V_th, scales.py)",
            "phi_eq": "phi de equilibrio sin iluminacion (phi_n = phi_p = 0)",
            "G": "generacion adimensional de dPV (cell.G)",
            "Jn,Jp": "(n_voltajes, N-1) corrientes de Scharfetter-Gummel adimensionales (current.py)",
            "seg_*": "segundos de reloj"})
        f.attrs["creado"] = time.strftime("%Y-%m-%d %H:%M:%S")
        for g in ("sims", "esquinas", "refinados"):
            f.create_group(g)
    return f


GRUPO_POR_TIPO = {"sobol": "sims", "esquina": "esquinas", "refinado": "refinados"}


def escribir(f, r):
    g = f[GRUPO_POR_TIPO[r["tipo"]]].create_group(f"{r['idx']:06d}")
    g.attrs["estado"] = r["estado"]
    g.attrs["mensaje"] = r["mensaje"]
    g.attrs["xi"] = r["xi"]
    g.attrs["plataforma"] = r["plataforma"]
    for k in ESCALARES:
        if k in r:
            g.attrs[k] = r[k]
    for k in ARREGLOS_SIEMPRE:
        if k in r:
            g.create_dataset(k, data=r[k])
    if r["estado"] == "ok":
        for k in PERFILES:
            if k in r:
                g.create_dataset(k, data=r[k], compression="gzip", compression_opts=4, shuffle=True)


def hechas(f):
    return {(t, int(k)) for t, gname in GRUPO_POR_TIPO.items() for k in f[gname].keys()}


def escribir_indice(f):
    """Tabla plana /indice con una fila por dispositivo Sobol (para particionar al entrenar)."""
    filas = sorted(f["sims"].keys())
    if "indice" in f:
        del f["indice"]
    gi = f.create_group("indice")
    if not filas:
        return
    gi.create_dataset("idx", data=np.array([int(k) for k in filas], dtype=np.int64))
    gi.create_dataset("estado", data=np.array([f["sims"][k].attrs["estado"] for k in filas], dtype="S16"))
    gi.create_dataset("u", data=np.stack([f["sims"][k]["u"][()] for k in filas]))
    for k in ("eff", "voc", "jsc", "ff", "pmax", "n_voltajes", "seg_total"):
        gi.create_dataset(k, data=np.array([f["sims"][s].attrs.get(k, np.nan) for s in filas], dtype=np.float64))


# -----------------------------------------------------------------------------
# Ejecucion de un lote de tareas con ventana acotada y presupuesto de tiempo
# -----------------------------------------------------------------------------
def ejecutar_tareas(tareas, plataforma, n_trabajadores, alrecibir, presupuesto_s=None, etiqueta=""):
    t0 = time.time()
    pendientes = list(tareas)[::-1]
    hechos, estados = 0, Counter()
    pool = crear_pool(plataforma, n_trabajadores)
    en_vuelo = {}
    try:
        while pendientes or en_vuelo:
            tiempo_ok = presupuesto_s is None or (time.time() - t0) < presupuesto_s
            while pendientes and tiempo_ok and len(en_vuelo) < 2 * n_trabajadores:
                t = pendientes.pop()
                en_vuelo[pool.submit(dpvgen.simular, t)] = t
            if not en_vuelo:
                break
            listos, _ = wait(list(en_vuelo), return_when=FIRST_COMPLETED)
            for fut in listos:
                t = en_vuelo.pop(fut)
                try:
                    r = fut.result()
                except BrokenProcessPool:
                    raise
                except Exception as e:  # el trabajador no devolvio nada
                    r = {**{k: t[k] for k in ("idx", "tipo", "familia", "etapa", "factor_malla")},
                         "u": np.asarray(t["u"]), "xi": json.dumps(t["xi"]), "estado": "error",
                         "mensaje": f"{type(e).__name__}: {e}", "plataforma": plataforma, "pid": -1,
                         "seg_total": float("nan")}
                alrecibir(r)
                hechos += 1
                estados[r["estado"]] += 1
                if hechos % 50 == 0 or hechos <= 3:
                    dt = time.time() - t0
                    print(f"[{etiqueta}] {hechos} hechas, {len(pendientes)} por lanzar | "
                          f"{dt/60:.1f} min | {3600*hechos/dt:.0f} sims/h | {dict(estados)}", flush=True)
    except BrokenProcessPool:
        # Un trabajador murio (memoria, segfault). Las tareas en vuelo se registran y se sigue.
        for fut, t in en_vuelo.items():
            alrecibir({**{k: t[k] for k in ("idx", "tipo", "familia", "etapa", "factor_malla")},
                       "u": np.asarray(t["u"]), "xi": json.dumps(t["xi"]), "estado": "error",
                       "mensaje": "proceso trabajador caido (BrokenProcessPool)",
                       "plataforma": plataforma, "pid": -1, "seg_total": float("nan")})
        pool.shutdown(wait=False, cancel_futures=True)
        print("Pool caido; se relanza con las tareas restantes.", flush=True)
        resto = presupuesto_s - (time.time() - t0) if presupuesto_s else None
        h2, e2 = ejecutar_tareas(pendientes[::-1], plataforma, n_trabajadores, alrecibir, resto, etiqueta)
        return hechos + h2, estados + e2
    pool.shutdown(wait=True)
    return hechos, estados


# -----------------------------------------------------------------------------
# PRODUCCION
# -----------------------------------------------------------------------------
def ejecutar_bloque(familia, etapa, bloque, presupuesto_s=None):
    """Produce un bloque de la secuencia Sobol fija de una familia-etapa en su propio .h5.
    Un bloque cortado por el presupuesto queda con lo hecho y se repite entero en otra
    sesion; al unir bloques, un indice repetido conserva la ultima corrida."""
    nombre = f"dpv_{familia}{etapa}_b{bloque:03d}.h5"
    ruta = os.path.join(C.DIR_SALIDA, nombre)
    fraccion = 0.0 if (C.REFINAR_SOLO_BLOQUE0 and bloque > 0) else C.FRACCION_REFINADO
    tareas = dpvgen.plan_de_bloque(familia, etapa, bloque, fraccion, dpvgen.TAM_BLOQUE)
    f = abrir_h5(ruta, familia, etapa, dpvgen.TAM_BLOQUE)
    f.attrs["bloque"] = bloque
    f.attrs["tam_bloque"] = dpvgen.TAM_BLOQUE
    f.attrs["indices_sobol"] = [bloque * dpvgen.TAM_BLOQUE, (bloque + 1) * dpvgen.TAM_BLOQUE]
    ya = hechas(f)
    tareas = [t for t in tareas if (t["tipo"], t["idx"]) not in ya]
    nw = C.N_TRABAJADORES or (os.cpu_count() if C.PLATAFORMA == "cpu" else max(1, n_gpus_visibles()))
    print(f"{nombre}: {len(tareas)} tareas ({dict(Counter(t['tipo'] for t in tareas))}), "
          f"plataforma={C.PLATAFORMA}, procesos={nw}", flush=True)

    n = [0]
    def alrecibir(r):
        escribir(f, r)
        n[0] += 1
        if n[0] % 100 == 0:
            f.flush()

    try:
        hechos, estados = ejecutar_tareas(tareas, C.PLATAFORMA, nw, alrecibir, presupuesto_s,
                                          f"{familia}{etapa}-b{bloque:03d}")
    finally:
        escribir_indice(f)
        f.attrs["actualizado"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["completo"] = hechos == len(tareas)
        f.close()
    print(f"{nombre}: {hechos}/{len(tareas)} simulaciones, estados {dict(estados)}, "
          f"{os.path.getsize(ruta)/1e6:.0f} MB" + ("" if hechos == len(tareas) else
          " -- INCOMPLETO: repetir este bloque en otra sesion"), flush=True)
    return {"unidad": f"{familia}{etapa}-b{bloque:03d}", "archivo": nombre, "hechas": hechos,
            "tareas": len(tareas), "completo": hechos == len(tareas), "estados": dict(estados)}


# -----------------------------------------------------------------------------
# Utilidad de los tests: un lote corto de tareas en una plataforma
# -----------------------------------------------------------------------------
def _uno(plataforma, tareas, n_trabajadores=1):
    res = []
    t0 = time.time()
    ejecutar_tareas(tareas, plataforma, n_trabajadores, res.append, None, f"test-{plataforma}")
    return res, time.time() - t0


# -----------------------------------------------------------------------------
# UNIDADES: lo que el runner de /remote-execution entrega en FORGE_RUN_UNITS
# -----------------------------------------------------------------------------
def parsear_unidad(u):
    """"E2B-b003" -> ("E2", "B", 3): familia, etapa y bloque de 1024 dispositivos."""
    import re
    m = re.fullmatch(r"(E[1-4])([AB])-b(\d{1,3})", u)
    if not m:
        raise ValueError(f"unidad invalida: {u!r} (se espera p.ej. 'E2B-b003')")
    return m.group(1), m.group(2), int(m.group(3))


def ejecutar_unidades(unidades):
    """Produce las unidades en orden dentro del presupuesto total de la sesion y deja
    resumen_sesion.json con lo que quedo completo y lo que hay que repetir."""
    # una unidad del runner es una sesion de Kaggle; puede empaquetar varios
    # bloques unidos con "+" ("E1A-b000+E1A-b001"), que se producen en orden
    unidades = [b for u in unidades for b in u.split("+")]
    t0 = time.time()
    hechas_u = []
    for u in unidades:
        fam, et, b = parsear_unidad(u)
        resto = C.PRESUPUESTO_HORAS * 3600 - (time.time() - t0)
        if resto <= 600:
            print(f"Sin tiempo para {u}; queda para otra sesion.", flush=True)
            hechas_u.append({"unidad": u, "completo": False, "hechas": 0, "motivo": "sin tiempo"})
            continue
        hechas_u.append(ejecutar_bloque(fam, et, b, resto))
    resumen = {"unidades": hechas_u, "horas": (time.time() - t0) / 3600,
               "commit": os.environ.get("FORGE_CLONE_COMMIT"), "plataforma": C.PLATAFORMA}
    with open(os.path.join(C.DIR_SALIDA, "resumen_sesion.json"), "w") as fh:
        json.dump(resumen, fh, indent=1)
    print(json.dumps(resumen, indent=1), flush=True)
    return resumen


# -----------------------------------------------------------------------------
# TEST RAPIDO: una plataforma por corrida, 20 variaciones, veredicto
# -----------------------------------------------------------------------------
def ejecutar_test_rapido(plataforma, n_variaciones=20):
    """Mide UNA plataforma: 4 nominales (correccion) y n_variaciones Sobol repartidas entre
    E1B-E4B, con todos los procesos que esa plataforma admite. Escribe test_<plataforma>.json
    y, si los nominales resuelven, shard.json (evidencia del ensayo)."""
    t0 = time.time()
    ng = n_gpus_visibles()
    if plataforma == "gpu" and ng == 0:
        raise RuntimeError("bandera 'gpu' pero la sesion no tiene GPU")
    nw = C.N_TRABAJADORES or (os.cpu_count() if plataforma == "cpu" else ng)
    print(f"TEST RAPIDO plataforma={plataforma} procesos={nw} | CPUs={os.cpu_count()} GPUs={ng}", flush=True)
    with crear_pool(plataforma, 1) as pool:
        info = pool.submit(dpvgen.info_trabajador).result()
    print("  jax:", info["jax"], "| backend:", info["backend"], "|", info["dispositivos"], flush=True)

    nominales = [{"idx": 0, "tipo": "sobol", "familia": fam, "etapa": "B", "u": [],
                  "xi": dpvgen.xi_nominal(fam), "factor_malla": 1} for fam in ("E1", "E2", "E3", "E4")]
    nom, _ = _uno(plataforma, nominales, min(nw, 4))
    for r in sorted(nom, key=lambda r: r["familia"]):
        print(f"  nominal {r['familia']}: {r['estado']:13s} PCE={100*r.get('eff', np.nan):.4f} % "
              f"Voc={r.get('voc', np.nan):.4f} V n_V={r.get('n_voltajes', 0)} t={r['seg_total']:.1f} s "
              f"{r['mensaje'][:80]}", flush=True)

    por_familia = n_variaciones // 4
    tareas = []
    for fam in ("E1", "E2", "E3", "E4"):
        tareas += dpvgen.plan_de_tareas(fam, "B", C.N_TOTAL, C.FRACCION_REFINADO)[:por_familia]
    t1 = time.time()
    res, dt = _uno(plataforma, tareas, nw)
    # La primera simulacion de cada proceso con cada tamano de malla incluye la compilacion
    # JIT de jax (E1/E3: 500 nodos uniformes; E2/E4: 180 de la malla de ex2); el resto es regimen.
    malla = {"E1": 500, "E3": 500, "E2": 180, "E4": 180}
    vistos, regimen = set(), {}
    for r in sorted(res, key=lambda r: r.get("t_fin", 0.0)):
        clave = (r["pid"], malla[r["familia"]])
        if clave in vistos:
            regimen.setdefault(r["familia"], []).append(r["seg_total"])
        vistos.add(clave)
    seg = {fam: float(np.median(v)) for fam, v in regimen.items() if v}
    sims_h = {fam: 3600 * nw / s for fam, s in seg.items()}
    estados = Counter(r["estado"] for r in res)
    sims_fe = C.N_TOTAL * (1 + C.FRACCION_REFINADO * 4) + 32
    horas = {f"{fam}B": sims_fe / v for fam, v in sims_h.items()}
    print(f"  {len(res)} variaciones en {dt:.0f} s de reloj ({3600*len(res)/dt:.0f} sims/h con JIT incluido)"
          f" | estados {dict(estados)}", flush=True)
    for fam in sorted(seg):
        print(f"  {fam}B: {seg[fam]:.2f} s/sim por proceso -> {sims_h[fam]:.0f} sims/h con {nw} procesos"
              f" -> ~{horas[fam + 'B']:.1f} h por familia-etapa", flush=True)

    fallos = [r["familia"] for r in nom if r["estado"] != "ok"]
    resumen = {"plataforma": plataforma, "procesos": nw, "cpus": os.cpu_count(), "gpus": ng,
               "jax": info, "nominales": {r["familia"]: {"estado": r["estado"], "eff": r.get("eff"),
                                                         "voc": r.get("voc"), "seg": r["seg_total"]} for r in nom},
               "estados_variaciones": dict(estados), "seg_por_sim_regimen": seg,
               "sims_por_hora": sims_h, "horas_por_familia_etapa": horas,
               "reloj_variaciones_s": dt, "reloj_total_s": time.time() - t0,
               "veredicto": "ok" if not fallos else f"fallan nominales {fallos}"}
    with open(os.path.join(C.DIR_SALIDA, f"test_{plataforma}.json"), "w") as fh:
        json.dump(resumen, fh, indent=1, default=float)
    if not fallos:
        with open(os.path.join(C.DIR_SALIDA, "shard.json"), "w") as fh:
            json.dump({"evidence": {"commit": os.environ.get("FORGE_CLONE_COMMIT"),
                                    "plataforma": plataforma}, "test": resumen}, fh, indent=1, default=float)
    print(f"VEREDICTO {plataforma}: {resumen['veredicto']}; tiempo total {resumen['reloj_total_s']/60:.1f} min",
          flush=True)
    return resumen
