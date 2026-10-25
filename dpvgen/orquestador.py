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
    PLATAFORMA="cpu", N_TRABAJADORES=None, PRESUPUESTO_HORAS=11.0,
    N_SHARDS=1, SHARD_ID=0, REANUDAR_DESDE=None,
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
def ejecutar_produccion():
    nombre = f"dpv_{C.FAMILIA}{C.ETAPA}_frag{C.SHARD_ID:02d}de{C.N_SHARDS:02d}.h5"
    ruta = os.path.join(C.DIR_SALIDA, nombre)
    if C.REANUDAR_DESDE and not os.path.exists(ruta):
        shutil.copy(C.REANUDAR_DESDE, ruta)
        print(f"Reanudando desde {C.REANUDAR_DESDE}")
    tareas = dpvgen.plan_de_tareas(C.FAMILIA, C.ETAPA, C.N_TOTAL, C.FRACCION_REFINADO)
    tareas = [t for i, t in enumerate(tareas) if i % C.N_SHARDS == C.SHARD_ID]
    f = abrir_h5(ruta, C.FAMILIA, C.ETAPA, C.N_TOTAL)
    ya = hechas(f)
    tareas = [t for t in tareas if (t["tipo"], t["idx"]) not in ya]
    nw = C.N_TRABAJADORES or (os.cpu_count() if C.PLATAFORMA == "cpu" else max(1, n_gpus_visibles()))
    print(f"{nombre}: {len(tareas)} tareas por hacer ({len(ya)} ya hechas), "
          f"plataforma={C.PLATAFORMA}, trabajadores={nw}")
    print(Counter(t["tipo"] for t in tareas))

    n = [0]
    def alrecibir(r):
        escribir(f, r)
        n[0] += 1
        if n[0] % 100 == 0:
            f.flush()

    try:
        hechos, estados = ejecutar_tareas(tareas, C.PLATAFORMA, nw, alrecibir,
                                          C.PRESUPUESTO_HORAS * 3600, f"{C.FAMILIA}{C.ETAPA}")
    finally:
        escribir_indice(f)
        f.attrs["actualizado"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.close()
    faltan = len(tareas) - hechos
    print(f"\nTerminado: {hechos} simulaciones en esta corrida, estados {dict(estados)}")
    print(f"Archivo: {ruta}  ({os.path.getsize(ruta)/1e9:.2f} GB)")
    if faltan > 0:
        print(f"QUEDAN {faltan} tareas: agrega este output como input de la siguiente corrida "
              f"y pon C.REANUDAR_DESDE en la ruta del .h5")
    return ruta


# -----------------------------------------------------------------------------
# TEST: correccion CPU vs GPU, tiempos, paralelismo, tamano y una mini-generacion
# -----------------------------------------------------------------------------
def _uno(plataforma, tareas, n_trabajadores=1):
    res = []
    t0 = time.time()
    ejecutar_tareas(tareas, plataforma, n_trabajadores, res.append, None, f"test-{plataforma}")
    return res, time.time() - t0


def ejecutar_test():
    print("=" * 78)
    print("1. ENTORNO")
    ng = n_gpus_visibles()
    print(f"CPUs: {os.cpu_count()} | GPUs visibles: {ng}")
    if ng:
        print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                             capture_output=True, text=True).stdout)
    plataformas = ["cpu"] + (["gpu"] if ng else [])
    for pl in plataformas:
        with crear_pool(pl, 1) as pool:
            print(pl, pool.submit(dpvgen.info_trabajador).result())

    print("=" * 78)
    print("2. CORRECCION: dispositivo nominal de cada familia, CPU frente a GPU")
    nominales = []
    for fam in ("E1", "E2", "E3", "E4"):
        nominales.append({"idx": 0, "tipo": "sobol", "familia": fam, "etapa": "B",
                          "u": [], "xi": dpvgen.xi_nominal(fam), "factor_malla": 1})
    ref = {}
    for pl in plataformas:
        res, dt = _uno(pl, nominales)
        for r in res:
            ref[(pl, r["familia"])] = r
            print(f"  {pl:3s} {r['familia']}: estado={r['estado']:13s} PCE={100*r.get('eff', np.nan):.5f} % "
                  f"Voc={r.get('voc', np.nan):.4f} V  Jsc={r.get('jsc', np.nan)/10:.3f} mA/cm2 "
                  f"FF={r.get('ff', np.nan):.4f}  n_V={r.get('n_voltajes', 0)}  t={r['seg_total']:.1f} s "
                  f"{r['mensaje'][:80]}")
    if "gpu" in plataformas:
        print("  Diferencias GPU - CPU:")
        for fam in ("E1", "E2", "E3", "E4"):
            a, b = ref[("cpu", fam)], ref[("gpu", fam)]
            if a["estado"] == b["estado"] == "ok" and a["pots"].shape == b["pots"].shape:
                print(f"    {fam}: |dPCE| = {abs(a['eff']-b['eff'])*100:.2e} pp, "
                      f"max|d pots| = {np.max(np.abs(a['pots']-b['pots'])):.2e} V_th, "
                      f"max|dJ|/|Jsc| = {np.max(np.abs(a['J']-b['J']))/abs(a['J'][0]):.2e}")
            else:
                print(f"    {fam}: no comparable (estados {a['estado']} / {b['estado']}, "
                      f"n_V {a.get('n_voltajes')} / {b.get('n_voltajes')})")
    e4 = ref[("cpu", "E4")]
    if e4["estado"] == "ok":
        print(f"  E4: P_in transformado = {e4['P_in_luz']:.2f} W/m2, denominador estandar = "
              f"{e4['P_in_std']:.2f} W/m2; PCE(Ec. 14) = {100*e4['eff']:.4f} %, "
              f"PCE que daria dPV = {100*e4['eff_dpv']:.4f} %")

    print("=" * 78)
    print(f"3. TIEMPO: {C.N_TEST_SIMS} dispositivos Sobol de {C.TEST_FAMILIA}{C.TEST_ETAPA}, en serie")
    plan = dpvgen.plan_de_tareas(C.TEST_FAMILIA, C.TEST_ETAPA, C.N_TOTAL, C.FRACCION_REFINADO)
    muestra = plan[:C.N_TEST_SIMS]
    serie = {}
    for pl in plataformas:
        res, dt = _uno(pl, muestra)
        ok = [r["seg_total"] for r in res if np.isfinite(r["seg_total"])]
        serie[pl] = float(np.median(ok[1:] if len(ok) > 1 else ok))
        print(f"  {pl}: primera (con JIT) {ok[0]:.1f} s, mediana del resto {serie[pl]:.2f} s/sim, "
              f"estados {Counter(r['estado'] for r in res)}")

    print("=" * 78)
    print("4. PARALELISMO: rendimiento con varios procesos a la vez")
    # Se mide en la familia de prueba (malla graduada, 180 nodos) y en E1B (500 nodos):
    # la GPU puede ganar o perder segun el tamano del sistema lineal.
    rendimiento = {}
    candidatos = [("cpu", w) for w in sorted({1, 2, max(1, os.cpu_count() // 2), os.cpu_count()})]
    if ng:
        candidatos += [("gpu", w) for w in sorted({ng, 2 * ng})]
    for fe in dict.fromkeys([C.TEST_FAMILIA + C.TEST_ETAPA, "E1B"]):
        plan_fe = dpvgen.plan_de_tareas(fe[:2], fe[2], C.N_TOTAL, C.FRACCION_REFINADO)
        for pl, w in candidatos:
            tareas = plan_fe[C.N_TEST_SIMS:C.N_TEST_SIMS + 3 * w]
            _uno(pl, tareas[:w], w)                        # calentar JIT en cada proceso
            res, dt = _uno(pl, tareas, w)
            rendimiento[(fe, pl, w)] = 3600 * len(res) / dt
            print(f"  {fe} {pl} x{w}: {rendimiento[(fe, pl, w)]:.0f} sims/h")
    total = {}
    for (fe, pl, w), v in rendimiento.items():
        total.setdefault((pl, w), []).append(v)
    # La configuracion elegida es la de mejor rendimiento en el peor caso de las familias medidas
    mejor = max(total, key=lambda k: min(total[k]))
    sims_fe = C.N_TOTAL * (1 + C.FRACCION_REFINADO * 4) + 32
    horas = {fe: sims_fe / rendimiento[(fe, *mejor)] for fe in dict.fromkeys(k[0] for k in rendimiento)}
    print(f"  MEJOR: PLATAFORMA='{mejor[0]}', N_TRABAJADORES={mejor[1]} -> horas estimadas por "
          f"familia-etapa (incluye refinados con 4x nodos como 4 sims): "
          + ", ".join(f"{fe} {h:.1f} h" for fe, h in horas.items()))

    print("=" * 78)
    print(f"5. MINI-GENERACION: {C.N_TEST_SIMS} dispositivos + 4 esquinas + 2 refinados de "
          f"{C.TEST_FAMILIA}{C.TEST_ETAPA}")
    ruta = os.path.join(C.DIR_SALIDA, f"prueba_{C.TEST_FAMILIA}{C.TEST_ETAPA}.h5")
    if os.path.exists(ruta):
        os.remove(ruta)
    f = abrir_h5(ruta, C.TEST_FAMILIA, C.TEST_ETAPA, C.N_TOTAL)
    mini = plan[:C.N_TEST_SIMS] + [t for t in plan if t["tipo"] == "esquina"][:4] \
        + [t for t in plan if t["tipo"] == "refinado"][:2]
    ejecutar_tareas(mini, mejor[0], mejor[1], lambda r: escribir(f, r), None, "mini")
    escribir_indice(f)
    f.close()
    estados_mini, bytes_ok = {}, float("nan")
    with h5py.File(ruta, "r") as f:
        for gname in ("sims", "esquinas", "refinados"):
            est = Counter(f[gname][k].attrs["estado"] for k in f[gname])
            estados_mini[gname] = dict(est)
            print(f"  /{gname}: {len(f[gname])} grupos, estados {dict(est)}")
        oks = [k for k in f["sims"] if f["sims"][k].attrs["estado"] == "ok"]
        if oks:
            g = f["sims"][oks[0]]
            print("  ejemplo /sims/" + oks[0] + ":", {k: g[k].shape for k in g})
            bytes_ok = os.path.getsize(ruta) / max(1, len(f["sims"]) + len(f["esquinas"]))
            print(f"  ~{bytes_ok/1e3:.0f} kB por dispositivo -> ~{bytes_ok*C.N_TOTAL/1e9:.1f} GB "
                  f"por familia-etapa (limite de /kaggle/working: 20 GB)")

    resumen = {
        "cpus": os.cpu_count(), "gpus": ng,
        "nominales": {f"{pl}-{fam}": {k: (float(r[k]) if k in r and np.isscalar(r[k]) else None)
                                      for k in ("eff", "voc", "jsc", "ff", "n_voltajes", "seg_total")}
                      | {"estado": r["estado"], "mensaje": r["mensaje"][:300]}
                      for (pl, fam), r in ref.items()},
        "diferencias_gpu_cpu": {
            fam: {"dPCE_pp": abs(ref[("cpu", fam)]["eff"] - ref[("gpu", fam)]["eff"]) * 100,
                  "max_dpots": float(np.max(np.abs(ref[("cpu", fam)]["pots"] - ref[("gpu", fam)]["pots"])))}
            for fam in ("E1", "E2", "E3", "E4")
            if "gpu" in plataformas and ref[("cpu", fam)]["estado"] == ref[("gpu", fam)]["estado"] == "ok"
            and ref[("cpu", fam)]["pots"].shape == ref[("gpu", fam)]["pots"].shape},
        "seg_por_sim_serie": serie,
        "rendimiento_sims_h": {f"{fe}|{pl}|{w}": v for (fe, pl, w), v in rendimiento.items()},
        "mejor": {"plataforma": mejor[0], "trabajadores": mejor[1]},
        "horas_por_familia_etapa": horas,
        "kB_por_dispositivo": bytes_ok / 1e3,
        "estados_mini": estados_mini,
    }
    with open(os.path.join(C.DIR_SALIDA, "test_resumen.json"), "w") as fh:
        json.dump(resumen, fh, indent=1, default=float)
    print("\nResumen escrito en test_resumen.json")

    # Evidencia del ensayo para `remote_cli smoke record`: solo se escribe si los cuatro
    # nominales resolvieron en todas las plataformas medidas. Un ensayo fallido no deja
    # shard.json, y sin el no hay veredicto "pass".
    fallos = [f"{pl}-{fam}" for (pl, fam), r in ref.items() if r["estado"] != "ok"]
    if fallos:
        print(f"ENSAYO FALLIDO: nominales sin estado ok: {fallos}. No se escribe shard.json.")
    else:
        stamp = {"evidence": {"commit": os.environ.get("FORGE_CLONE_COMMIT"),
                              "nominalesOk": sorted(f"{pl}-{fam}" for pl, fam in ref)},
                 "environment": {"cpus": os.cpu_count(), "gpus": ng},
                 "test": resumen}
        with open(os.path.join(C.DIR_SALIDA, "shard.json"), "w") as fh:
            json.dump(stamp, fh, indent=1, default=float)
        print("Ensayo correcto: shard.json escrito.")
    return resumen


# -----------------------------------------------------------------------------
# UNIDADES: lo que el runner de /remote-execution entrega en FORGE_RUN_UNITS
# -----------------------------------------------------------------------------
def parsear_unidad(u):
    """"E2B-f01de04" -> ("E2", "B", 1, 4); "E2B" -> ("E2", "B", 0, 1)."""
    import re
    m = re.fullmatch(r"(E[1-4])([AB])(?:-f(\d+)de(\d+))?", u)
    if not m:
        raise ValueError(f"unidad invalida: {u!r} (se espera p.ej. 'E2B' o 'E2B-f01de04')")
    return m.group(1), m.group(2), int(m.group(3) or 0), int(m.group(4) or 1)


def ejecutar_unidades(unidades):
    """Produce una unidad tras otra dentro del presupuesto total de la sesion."""
    t0 = time.time()
    rutas = []
    total = C.PRESUPUESTO_HORAS
    for u in unidades:
        fam, et, sid, ns = parsear_unidad(u)
        resto = total - (time.time() - t0) / 3600
        if resto <= 0.25:
            print(f"Sin tiempo para {u}; queda para otra sesion.")
            break
        configurar(FAMILIA=fam, ETAPA=et, SHARD_ID=sid, N_SHARDS=ns, PRESUPUESTO_HORAS=resto)
        rutas.append(ejecutar_produccion())
    C.PRESUPUESTO_HORAS = total
    return rutas
