"""Generador del dataset sintetico de dPV para el benchmark PINN (propuesta r04, Seccion 7).

Este modulo se importa en cada proceso trabajador. No importa jax ni deltapv al
cargarse: `init_trabajador` fija primero la plataforma (CPU o GPU) y los hilos, y
solo despues importa deltapv, porque jax lee esas variables al inicializarse.
"""
import os
import re
import json
import math
import time
import logging
import traceback

import numpy as np

DPV = None          # modulo deltapv, cargado en init_trabajador
_CAPTURA = None     # manejador que guarda los mensajes del logger "deltapv"
PLATAFORMA = None

TOL_NEWTON = 1e-6   # criterio de parada de solver.solve en dPV (|p| <= 1e-6)
MAX_ITER_NEWTON = 100

# =============================================================================
# Familias, parametros y rangos (r04, Seccion 7)
#   escala "log": se muestrea uniforme en log10 (Ec. 3); "lin": uniforme lineal.
#   etapa "A": entra en xi desde la etapa A; "B": solo en la etapa B (en A queda
#   fijo en el nominal). Espesores en cm (unidad de dPV); dopajes en cm^-3.
# =============================================================================
FAMILIAS = {
    "E1": {
        "descripcion": "Homounion p-n (examples/ex1_np_junction.py)",
        "parametros": [
            # nombre, escala, minimo, maximo, nominal, etapa, fuente
            ("Nd_n", "log", 1e15, 8e17, 1e17, "A", "ex1 menos dos decadas; cota superior Nc = 8e17"),
            ("Na_p", "log", 1e15, 1e18, 1e17, "A", "ex1 +- una a dos decadas; por debajo de Nv = 1.8e19"),
            ("L_n", "lin", 0.1e-4, 3e-4, 1e-4, "B", "eleccion declarada, sin fuente (r04)"),
            ("L_p", "lin", 0.1e-4, 3e-4, 1e-4, "B", "eleccion declarada, sin fuente (r04)"),
        ],
    },
    "E2": {
        "descripcion": "Heterounion CdS/CdTe (examples/ex2_np_hetero.py)",
        "parametros": [
            ("Nd_CdS", "log", 1e16, 1e18, 1e17, "A", "ex2 +- una decada; por debajo de Nc = 2.2e18"),
            ("Na_CdTe", "log", 1e14, 1e17, 1e15, "A", "Metzger et al. 2019, Nat. Energy 4, 837"),
            ("L_CdS", "lin", 25e-7, 100e-7, 25e-7, "B", "lilhare2020: dispositivos de 40 a 110 nm; cota inferior el nominal de ex2"),
            ("L_CdTe", "lin", 1e-4, 4e-4, 4e-4, "B", "Mykytyuk et al. 2012, Acta Phys. Pol. A 122, 1073"),
        ],
    },
    "E3": {
        "descripcion": "Perovskita ETM/absorbedor/HTM (optimize/psc.py, diseno x0)",
        "parametros": [
            ("Nd_ETM", "log", 1e17, 6.8e18, 4.4e18, "A", "cota inferior de psc.py; cota superior Nc del ETM (6.83e18)"),
            # r04 escribe 1.9e18, pero Nv del HTM de x0 es 10**18.2718 = 1.87e18; se usa Nv
            # para respetar la regla de no degeneracion de la propia r04.
            ("Na_HTM", "log", 1e17, 1.8698e18, 4.4e17, "A", "cota inferior de psc.py; cota superior Nv del HTM (1.87e18)"),
            ("L_abs", "lin", 170e-7, 650e-7, 350e-7, "B", "Perovskite Database v1.0.1, p5 a p95"),
            ("L_ETL", "lin", 20e-7, 550e-7, 60e-7, "B", "Perovskite Database v1.0.1, p5 a p95"),
            ("L_HTL", "lin", 15e-7, 300e-7, 100e-7, "B", "Perovskite Database v1.0.1, p5 a p95"),
        ],
    },
    "E4": {
        "descripcion": "CdS/CdTe con capa LDS como transformacion espectral (Ec. 9)",
        "parametros": [
            ("Nd_CdS", "log", 1e16, 1e18, 1e17, "A", "igual que E2"),
            ("Na_CdTe", "log", 1e14, 1e17, 1e15, "A", "igual que E2"),
            ("eta_PL", "lin", 0.0, 0.9, 0.8, "A", "kalytchuk2014; Solodovnyk et al. 2016 (cota 0.9)"),
            ("L_CdS", "lin", 25e-7, 100e-7, 25e-7, "B", "igual que E2"),
            ("L_CdTe", "lin", 1e-4, 4e-4, 4e-4, "B", "igual que E2"),
        ],
    },
}

# Semilla de Sobol fija por familia y etapa: el dataset es uno solo. La semilla de
# cada corrida de entrenamiento cambia la particion, no los dispositivos.
SEMILLAS_SOBOL = {"E1A": 1101, "E1B": 1102, "E2A": 1201, "E2B": 1202,
                  "E3A": 1301, "E3B": 1302, "E4A": 1401, "E4B": 1402}

# Capa LDS (r04 Ec. 9, kalytchuk2014)
LDS_A_UV = 0.95
LDS_VENTANA_NM = (300.0, 400.0)
LDS_ETA_GEOM = 0.87
LDS_LAMBDA_EM_NM = 606.5


def parametros_activos(familia, etapa):
    """Parametros que forman xi en esa etapa, en el orden declarado."""
    return [p for p in FAMILIAS[familia]["parametros"] if etapa == "B" or p[5] == "A"]


def xi_desde_unitario(familia, etapa, u):
    """Inversa de (Ec. 3): u en [0,1]^d -> valores fisicos (dict con TODOS los parametros)."""
    valores = {p[0]: p[4] for p in FAMILIAS[familia]["parametros"]}
    for uj, (nombre, escala, lo, hi, _nom, _et, _f) in zip(u, parametros_activos(familia, etapa)):
        if escala == "log":
            valores[nombre] = 10.0 ** (math.log10(lo) + float(uj) * (math.log10(hi) - math.log10(lo)))
        else:
            valores[nombre] = lo + float(uj) * (hi - lo)
    return valores


def xi_nominal(familia):
    return {p[0]: p[4] for p in FAMILIAS[familia]["parametros"]}


def plan_de_tareas(familia, etapa, n_total, fraccion_refinado=0.05):
    """Todas las tareas de una familia-etapa, deterministas dada la semilla de Sobol.

    tipo "sobol":    n_total dispositivos de Sobol aleatorizado (indices 0..n_total-1)
    tipo "esquina":  las 2^d esquinas del hipercubo (prueba de extrapolacion, r04 Sec. 7)
    tipo "refinado": 5 % de los dispositivos Sobol re-simulados con 4x nodos (Ec. 21)
    """
    from scipy.stats import qmc
    act = parametros_activos(familia, etapa)
    d = len(act)
    semilla = SEMILLAS_SOBOL[familia + etapa]
    m = int(math.ceil(math.log2(n_total)))
    u = qmc.Sobol(d=d, scramble=True, seed=semilla).random_base2(m=m)[:n_total]
    tareas = []
    for i in range(n_total):
        tareas.append({"idx": i, "tipo": "sobol", "u": u[i].tolist(), "factor_malla": 1})
    for c in range(2 ** d):
        uc = [float((c >> j) & 1) for j in range(d)]
        tareas.append({"idx": n_total + c, "tipo": "esquina", "u": uc, "factor_malla": 1})
    rng = np.random.default_rng(semilla + 7)
    n_ref = int(round(fraccion_refinado * n_total))
    for i in sorted(rng.choice(n_total, size=n_ref, replace=False).tolist()):
        tareas.append({"idx": i, "tipo": "refinado", "u": u[i].tolist(), "factor_malla": 4})
    for t in tareas:
        t["familia"], t["etapa"] = familia, etapa
        t["xi"] = xi_desde_unitario(familia, etapa, t["u"])
    return tareas


# =============================================================================
# Proceso trabajador
# =============================================================================
class _Captura(logging.Handler):
    """Guarda (tiempo, nivel, mensaje) de cada registro del logger de dPV.

    dPV no lanza ningun error cuando Newton agota sus 100 iteraciones: devuelve la
    ultima iteracion en silencio. La unica evidencia de convergencia son estos
    mensajes, por eso se capturan y se analizan por voltaje.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.eventos = []

    def emit(self, r):
        self.eventos.append((r.created, r.levelno, r.getMessage()))


def init_trabajador(plataforma="cpu", n_gpus=0, contador=None, hilos=1):
    global DPV, _CAPTURA, PLATAFORMA
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(hilos)
    if plataforma == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                                   f"intra_op_parallelism_threads={hilos}")
    else:
        os.environ["JAX_PLATFORMS"] = "cuda"
        if contador is not None and n_gpus > 0:
            with contador.get_lock():
                k = contador.value
                contador.value += 1
            os.environ["CUDA_VISIBLE_DEVICES"] = str(k % n_gpus)
    PLATAFORMA = plataforma
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # deltapv imprime su banner ASCII en cada import; con un proceso por nucleo eso
    # inunda la salida del cuaderno hasta que Kaggle no puede guardarlo. Se descarta.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        import deltapv as dpv
        import deltapv.current  # noqa: F401  (Jn, Jp)
    DPV = dpv
    lg = logging.getLogger("deltapv")
    _CAPTURA = _Captura()
    lg.handlers = [_CAPTURA]
    lg.propagate = False
    lg.setLevel(logging.INFO)


def info_trabajador():
    import jax
    import deltapv
    return {"jax": jax.__version__, "backend": jax.default_backend(),
            "dispositivos": [str(d) for d in jax.devices()],
            "x64": bool(jax.config.jax_enable_x64), "deltapv": deltapv.__file__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid()}


# -----------------------------------------------------------------------------
# Construccion de los dispositivos
# -----------------------------------------------------------------------------
def _malla_ex2(t1, t2, factor=1):
    """Malla graduada de examples/ex2_np_hetero.py, refinada en contactos e interfaz.

    ex2 la define (180 nodos) pero no se la pasa a make_design; r04 la adopta.
    Con factor=4 cada tramo tiene 4 veces mas nodos (piso de discretizacion, Ec. 21).
    """
    dd = 1e-7
    L = t1 + t2
    tramos = [(0, dd, 10, False), (dd, t1 - dd, 50, False), (t1 - dd, t1 + dd, 10, False),
              (t1 + dd, L - dd, 100, False), (L - dd, L, 10, True)]
    return np.concatenate([np.linspace(a, b, n * factor, endpoint=e) for a, b, n, e in tramos])


def _material_e1():
    return DPV.create_material(Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100,
                               Et=0, tn=1e-8, tp=1e-8, A=2e4)


def _materiales_cds_cdte():
    CdS = DPV.create_material(Nc=2.2e18, Nv=1.8e19, Eg=2.4, eps=10, Et=0, mn=100, mp=25,
                              tn=1e-8, tp=1e-13, Chi=4.0, A=1e4)
    CdTe = DPV.create_material(Nc=8e17, Nv=1.8e19, Eg=1.5, eps=9.4, Et=0, mn=320, mp=40,
                               tn=5e-9, tp=5e-9, Chi=3.9, A=1e4)
    return CdS, CdTe


# Diseno inicial x0 de optimize/psc.py (ETM y HTM no son materiales reales; r04 Sec. 7)
_X0_PSC = [1.661788237392516, 4.698293002285373, 19.6342803183675, 18.83471869026531,
           19.54569869328745, 0.7252792557586427, 1.6231392299175988,
           2.5268524699070234, 2.51936429069554, 6.933634938056497, 19.41835918276137,
           18.271793488422656, 0.46319949214386513, 0.2058139980642224,
           18.63975340175838, 17.643726318153238]


def _materiales_psc():
    x, tau, A = _X0_PSC, 1e-6, 2e4
    perov = DPV.create_material(Eg=1.5, Chi=3.9, eps=10, Nc=3.9e18, Nv=2.7e18, mn=2, mp=2,
                                tn=tau, tp=tau, Br=2.3e-9, A=A)
    etm = DPV.create_material(Eg=x[0], Chi=x[1], eps=x[2], Nc=10 ** x[3], Nv=10 ** x[4],
                              mn=10 ** x[5], mp=10 ** x[6], tn=tau, tp=tau, A=A)
    htm = DPV.create_material(Eg=x[7], Chi=x[8], eps=x[9], Nc=10 ** x[10], Nv=10 ** x[11],
                              mn=10 ** x[12], mp=10 ** x[13], tn=tau, tp=tau, A=A)
    return etm, perov, htm


def espectro_estandar():
    return DPV.incident_light("sun")


def espectro_lds(eta_pl):
    """(Ec. 9): la capa absorbe A_k del flujo fotonico en 300-400 nm y reemite en 606.5 nm.

    Se construye LightSource directamente: incident_light("user") reescala a 1000 W/m2
    y borraria la ganancia o perdida de la capa.
    """
    import deltapv.sun as sun
    lam = np.asarray(sun.Lambda_eff, dtype=np.float64)
    P = np.asarray(sun.P_in_eff, dtype=np.float64)
    A = np.where((lam >= LDS_VENTANA_NM[0]) & (lam <= LDS_VENTANA_NM[1]), LDS_A_UV, 0.0)
    k_em = int(np.argmin(np.abs(lam - LDS_LAMBDA_EM_NM)))
    fotones = P * lam                     # proporcional a Phi_k = P_k lambda_k / (hc)
    absorbidos = float(np.sum(A * fotones))
    fot_eff = fotones * (1.0 - A)
    fot_eff[k_em] += LDS_ETA_GEOM * float(eta_pl) * absorbidos
    P_eff = fot_eff / lam
    import jax.numpy as jnp
    return DPV.simulator.LightSource(Lambda=jnp.asarray(lam), P_in=jnp.asarray(P_eff))


def construir(familia, xi, factor_malla=1):
    """Devuelve (design, light) de dPV para la familia y los valores fisicos xi."""
    dpv = DPV
    if familia == "E1":
        mat = _material_e1()
        des = dpv.make_design(n_points=500 * factor_malla, Ls=[xi["L_n"], xi["L_p"]], mats=mat,
                              Ns=[xi["Nd_n"], -xi["Na_p"]],
                              Snl=1e7, Snr=0, Spl=0, Spr=1e7)          # contactos de ex1
        return des, espectro_estandar()
    if familia in ("E2", "E4"):
        CdS, CdTe = _materiales_cds_cdte()
        t1, t2 = xi["L_CdS"], xi["L_CdTe"]
        import jax.numpy as jnp
        grid = jnp.asarray(_malla_ex2(t1, t2, factor_malla))
        des = dpv.make_design(n_points=int(grid.size), Ls=[t1, t2], mats=[CdS, CdTe],
                              Ns=[xi["Nd_CdS"], -xi["Na_CdTe"]], grid=grid,
                              Snl=1.16e7, Snr=1.16e7, Spl=1.16e7, Spr=1.16e7)
        ls = espectro_estandar() if familia == "E2" else espectro_lds(xi["eta_PL"])
        return des, ls
    if familia == "E3":
        etm, perov, htm = _materiales_psc()
        des = dpv.make_design(n_points=500 * factor_malla,
                              Ls=[xi["L_ETL"], xi["L_abs"], xi["L_HTL"]], mats=[etm, perov, htm],
                              Ns=[xi["Nd_ETM"], 0, -xi["Na_HTM"]],
                              Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7)
        return des, espectro_estandar()
    raise ValueError(f"familia desconocida: {familia}")


# -----------------------------------------------------------------------------
# Analisis de los mensajes del solucionador
# -----------------------------------------------------------------------------
_RE_PASO = re.compile(r"Solving for\s+(-?[\d.]+) V \(Step\s+(\d+)\)")
_RE_ITER = re.compile(r"iteration\s+(\d+)\s+\|p\| = (\S+)\s+\|F\| = (\S+)")


def _nuevo_bloque():
    return {"niter": 0, "p": float("nan"), "F": float("nan"), "denso": False,
            "niter_aux": 0, "t0": None}


def _analizar_log(eventos, t_fin):
    eq = _nuevo_bloque()
    pasos = []
    actual, aux = eq, False
    for t, _nivel, msg in eventos:
        if actual["t0"] is None:
            actual["t0"] = t
        m = _RE_PASO.search(msg)
        if m:
            actual = _nuevo_bloque()
            actual["t0"] = t
            pasos.append(actual)
            aux = False
            continue
        if "for convergence" in msg:      # paso auxiliar de dPV en vstep == 1
            aux = True
            continue
        if msg.startswith("Continuing"):
            aux = False
            continue
        m = _RE_ITER.search(msg)
        if m:
            if aux:
                actual["niter_aux"] += 1
            else:
                actual["niter"] += 1
                actual["p"], actual["F"] = float(m.group(2)), float(m.group(3))
            continue
        if "Switching to dense" in msg:
            actual["denso"] = True
    bloques = [eq] + pasos
    for b, sig in zip(bloques, bloques[1:] + [None]):
        t1 = sig["t0"] if sig is not None else t_fin
        b["seg"] = (t1 - b["t0"]) if b["t0"] is not None else float("nan")
        # dPV itera mientras |p| > 1e-6; el formato %.2e redondea, de ahi la holgura
        b["convergio"] = bool(np.isfinite(b["p"]) and b["p"] <= TOL_NEWTON * 1.005)
    return eq, pasos


def _ultimo_mensaje_critico(eventos):
    for _t, nivel, msg in reversed(eventos):
        if nivel >= logging.ERROR:
            return msg.strip()
    return ""


# -----------------------------------------------------------------------------
# Una simulacion
# -----------------------------------------------------------------------------
def simular(tarea):
    """Simula un dispositivo y devuelve un dict serializable (arreglos numpy).

    estado: "ok" | "no_convergido" | "sin_cruce" | "nan" | "fallo_newton" | "error"
    Solo "ok" se usa para entrenar; los demas se guardan con sus parametros y su
    mensaje y su tasa se reporta (r04 Sec. 7: nunca se descartan en silencio).
    """
    dpv = DPV
    _CAPTURA.eventos = []
    t_ini = time.time()
    salida = {"idx": tarea["idx"], "tipo": tarea["tipo"], "familia": tarea["familia"],
              "etapa": tarea["etapa"], "u": np.asarray(tarea["u"], dtype=np.float64),
              "xi": json.dumps(tarea["xi"]), "factor_malla": tarea["factor_malla"],
              "estado": "ok", "mensaje": "", "plataforma": PLATAFORMA, "pid": os.getpid()}
    try:
        des, ls = construir(tarea["familia"], tarea["xi"], tarea["factor_malla"])
        res = dpv.simulate(des, ls, verbose=True)
        t_fin = time.time()
        eq, pasos = _analizar_log(_CAPTURA.eventos, t_fin)

        V = np.asarray(res["iv"][0], dtype=np.float64)          # V
        J = np.asarray(res["iv"][1], dtype=np.float64)          # A/cm2
        pmax = float(res["mpp"])                                  # W/m2
        P_std = float(np.sum(np.asarray(espectro_estandar().P_in)))
        P_luz = float(np.sum(np.asarray(ls.P_in)))
        # (Ec. 14): denominador = espectro estandar sin LDS (~899.9 W/m2), tambien en E4.
        # En E1-E3 coincide con res["eff"]; en E4 dPV dividiria por el espectro transformado.
        eff = pmax / P_std
        jsc = J[0] * 1e4                                          # A/m2
        voc = float("nan")
        if J.size >= 2 and J[-1] * J[-2] <= 0 and J[-2] != J[-1]:
            voc = float(V[-2] + J[-2] * (V[-1] - V[-2]) / (J[-2] - J[-1]))
        ff = pmax / (voc * jsc) if np.isfinite(voc) and jsc > 0 else float("nan")

        salida.update({
            "V": V, "J": J, "pmax": pmax, "vmax": float(res["vmax"]), "eff": eff,
            "eff_dpv": float(res["eff"]), "P_in_std": P_std, "P_in_luz": P_luz,
            "jsc": jsc, "voc": voc, "ff": ff, "n_voltajes": int(V.size),
            "newton_niter": np.array([b["niter"] for b in pasos], dtype=np.int32),
            "newton_niter_aux": np.array([b["niter_aux"] for b in pasos], dtype=np.int32),
            "newton_p": np.array([b["p"] for b in pasos]),
            "newton_F": np.array([b["F"] for b in pasos]),
            "newton_denso": np.array([b["denso"] for b in pasos], dtype=bool),
            "newton_convergio": np.array([b["convergio"] for b in pasos], dtype=bool),
            "seg_por_voltaje": np.array([b["seg"] for b in pasos]),
            "eq_niter": eq["niter"], "eq_p": eq["p"], "eq_convergio": eq["convergio"],
            "seg_equilibrio": eq["seg"],
        })

        guardar_perfiles = tarea["factor_malla"] == 1
        if guardar_perfiles:
            cell = res["cell"]
            pots = res["pots"]
            salida["x_cm"] = np.asarray(des.grid, dtype=np.float64)
            salida["G"] = np.asarray(cell.G, dtype=np.float64)
            salida["phi_eq"] = np.asarray(res["eq"].phi, dtype=np.float64)
            salida["pots"] = np.stack([np.stack([np.asarray(p.phi), np.asarray(p.phi_n),
                                                 np.asarray(p.phi_p)]) for p in pots]).astype(np.float64)
            salida["Jn"] = np.stack([np.asarray(dpv.current.Jn(cell, p)) for p in pots]).astype(np.float64)
            salida["Jp"] = np.stack([np.asarray(dpv.current.Jp(cell, p)) for p in pots]).astype(np.float64)

        finitos = [V, J, np.array([pmax])]
        if guardar_perfiles:
            finitos += [salida["pots"], salida["Jn"], salida["Jp"]]
        if not all(np.all(np.isfinite(a)) for a in finitos):
            salida["estado"], salida["mensaje"] = "nan", "valores no finitos en la solucion"
        elif not (eq["convergio"] and all(b["convergio"] for b in pasos)):
            malos = [i for i, b in enumerate(pasos) if not b["convergio"]]
            salida["estado"] = "no_convergido"
            salida["mensaje"] = (f"Newton sin converger (|p| > {TOL_NEWTON}) en equilibrio={not eq['convergio']}"
                                 f" y voltajes {malos}")
        elif not np.isfinite(voc):
            salida["estado"] = "sin_cruce"
            salida["mensaje"] = f"la curva J-V no cruza cero en {V.size} voltajes (limite de dPV: 100)"
    except SystemExit:
        # dPV hace `raise SystemExit` cuando el solucionador denso da NaN
        salida["estado"] = "fallo_newton"
        salida["mensaje"] = _ultimo_mensaje_critico(_CAPTURA.eventos) or "SystemExit en dPV"
    except Exception as e:  # noqa: BLE001
        salida["estado"] = "error"
        salida["mensaje"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    salida["seg_total"] = time.time() - t_ini
    salida["t_fin"] = time.time()
    return salida
