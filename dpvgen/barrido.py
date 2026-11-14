"""Discrete screening sweep: which design and material ranges give high efficiency (local CPU, JAX).

Structure: the E1 p-n homojunction of examples/ex1_np_junction.py with its contacts (Snl = Spr = 1e7 cm/s,
Snr = Spl = 0), the same 500-node mesh and the same AM1.5 spectrum. Only the absorber material changes.

Per material, fixed: Chi, eps, Nc, Nv, Br, Cn, Cp and the absorption prefactor A (taken from deltapv's own
material library `deltapv/resources/*.yaml` or from the deltapv examples). Chi is fixed on purpose: in a
homojunction it shifts both bands equally and the contacts are set by surface recombination, so it cannot move
the efficiency.

Per material, swept on discrete levels: Eg (offset from the material's value), the mobilities (one factor on
both mn and mp), the lifetimes (tn = tp), the dopings Nd_n and Na_p, and the thicknesses L_n and L_p.

Absorption: deltapv's default optics (`compute_G(..., optics=True)`) use alpha = A sqrt(E - Eg), so moving Eg
moves the absorption edge with it. The tabulated alpha of the library is not used by `simulate`.

Sampling: a scrambled Sobol sequence in [0,1]^7 per material, each coordinate mapped to a level index, so the
levels stay discrete and balanced. Derived for the analysis: the hole diffusion length
L_D = sqrt(mp kT/q tp) and the ratio L_n / L_D.
"""
import math

import numpy as np

KT_Q = 0.025852  # V at 300 K

# Values from deltapv: resources/<name>.yaml (GaAs, InP, Si) and the examples (E1 generic: ex1;
# CdTe: ex2; perovskite: optimize/psc.py). A = 2e4 where the library leaves it null or omits it.
MATERIALES = {
    "E1gen": dict(Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, Br=0.0, A=2e4,
                  fuente="deltapv examples/ex1_np_junction.py"),
    "CdTe": dict(Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=320, mp=40, Br=0.0, A=1e4,
                 fuente="deltapv examples/ex2_np_hetero.py"),
    "Perov": dict(Chi=3.9, Eg=1.5, eps=10, Nc=3.9e18, Nv=2.7e18, mn=2, mp=2, Br=2.3e-9, A=2e4,
                  fuente="deltapv optimize/psc.py"),
    "GaAs": dict(Chi=4.07, Eg=1.424, eps=12.9, Nc=4.7e17, Nv=9.0e18, mn=8500, mp=10, Br=7.2e-10,
                 Cn=1e-30, Cp=1e-30, A=2e4, fuente="deltapv resources/GaAs.yaml (A null -> 2e4)"),
    "InP": dict(Chi=4.38, Eg=1.344, eps=12.5, Nc=5.7e17, Nv=1.1e19, mn=5400, mp=200, Br=1.2e-10,
                Cn=9e-31, Cp=9e-31, A=2e4, fuente="deltapv resources/InP.yaml"),
    "Si": dict(Chi=4.05, Eg=1.12, eps=11.7, Nc=3.2e19, Nv=1.8e19, mn=1400, mp=450, Br=1.1e-14,
               Cn=1.1e-30, Cp=3e-31, A=2e4,
               fuente="deltapv resources/Si.yaml; indirect gap: the sqrt model overstates absorption"),
}

# Discrete levels. Dopings are capped per material at 0.8 Nc (n side) and 0.8 Nv (p side): non-degenerate.
NIVELES = {
    "dEg": [-0.2, -0.1, 0.0, 0.1, 0.2],             # eV, added to the material's Eg
    "f_mu": [0.1, 0.3, 1.0],                        # factor on mn and mp
    "tau": [1e-9, 1e-8, 1e-7, 1e-6],                # s, tn = tp
    "Nd_n": [1e15, 3e15, 1e16, 3e16, 1e17, 3e17, 1e18],   # cm^-3, half-decade steps
    "Na_p": [1e15, 3e15, 1e16, 3e16, 1e17, 3e17, 1e18],   # cm^-3, half-decade steps
    "L_n": [0.25e-4, 0.5e-4, 1e-4, 1.5e-4, 2e-4, 3e-4],   # cm
    "L_p": [0.5e-4, 1e-4, 2e-4],                    # cm
}
EJES = list(NIVELES)
N_POR_MATERIAL = 512
SEMILLA = 2611


def niveles(material):
    m = MATERIALES[material]
    lv = dict(NIVELES)
    lv["Nd_n"] = [v for v in NIVELES["Nd_n"] if v <= 0.8 * m["Nc"]]
    lv["Na_p"] = [v for v in NIVELES["Na_p"] if v <= 0.8 * m["Nv"]]
    return lv


def plan(material, n=N_POR_MATERIAL, inicio=0):
    """Tasks [inicio, inicio + n) of the fixed Sobol sequence of `material`, mapped to discrete levels."""
    from scipy.stats import qmc
    k = list(MATERIALES).index(material)
    m = int(math.ceil(math.log2(inicio + n)))
    u = qmc.Sobol(d=len(EJES), scramble=True, seed=SEMILLA + k).random_base2(m=m)[inicio:inicio + n]
    lv = niveles(material)
    tareas = []
    for i, ui in enumerate(u):
        xi = {"material": material}
        for e, uj in zip(EJES, ui):
            xi[e] = lv[e][min(int(uj * len(lv[e])), len(lv[e]) - 1)]
        tareas.append({"idx": inicio + i, "tipo": "barrido", "familia": "BAR", "etapa": material,
                       "u": ui.tolist(), "factor_malla": 1, "xi": xi})
    return tareas


def derivados(xi):
    m = MATERIALES[xi["material"]]
    mp = m["mp"] * xi["f_mu"]
    L_D = math.sqrt(mp * KT_Q * xi["tau"])          # cm
    return {"Eg": m["Eg"] + xi["dEg"], "mn": m["mn"] * xi["f_mu"], "mp": mp, "L_D": L_D,
            "Ln_LD": xi["L_n"] / L_D}


def construir_barrido(dpv, xi, factor_malla=1):
    m = {k: v for k, v in MATERIALES[xi["material"]].items() if k != "fuente"}
    m["Eg"] = m["Eg"] + xi["dEg"]
    m["mn"], m["mp"] = m["mn"] * xi["f_mu"], m["mp"] * xi["f_mu"]
    mat = dpv.create_material(Et=0, tn=xi["tau"], tp=xi["tau"], **m)
    des = dpv.make_design(n_points=500 * factor_malla, Ls=[xi["L_n"], xi["L_p"]], mats=mat,
                          Ns=[xi["Nd_n"], -xi["Na_p"]], Snl=1e7, Snr=0, Spl=0, Spr=1e7)
    return des, dpv.incident_light("sun")
