"""Build the screening-sweep analysis page (Plotly) from salidas/barrido150/analisis.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RAIZ = Path(__file__).resolve().parents[1]
src = RAIZ / "salidas/barrido150/analisis.json"
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ.parent / "analysis/barrido-cruces.html"

d = pd.DataFrame(json.loads(src.read_text()))
d["ok"] = d.estado == "ok"
d["eff"] = d.eff * 100
o = d[d.ok & (d.eff > 0) & d.kappa0.notna()].copy()
o["kappa0"] = o.kappa0.clip(lower=1e-4)

REG = [(0, 0.15, "κ₀ ≤ 0.15 · zona útil"), (0.15, 0.47, "0.15–0.47 · transición"), (0.47, 1e9, "κ₀ ≥ 0.47 · problemático")]
o["regimen"] = pd.cut(o.kappa0, [r[0] for r in REG] + [1e9], labels=[r[2] for r in REG], right=True)
MATS = list(dict.fromkeys(d.material))
VARS = ["Eg", "f_mu", "tau", "Nd_n", "Na_p", "L_n", "L_p", "L_D", "Ln_LD"]
NOMBRE = {"Eg": "E_g", "f_mu": "factor μ", "tau": "τ", "Nd_n": "N_d", "Na_p": "N_a", "L_n": "L_n",
          "L_p": "L_p", "L_D": "L_D", "Ln_LD": "L_n/L_D", "eff": "PCE", "kappa0": "κ₀"}


def r(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


# per-material table
tabla = []
for m in MATS:
    a, b = d[d.material == m], o[o.material == m]
    tabla.append({"m": m, "n": len(a), "conv": r(a.ok.mean() * 100, 0), "eff": r(b.eff.median(), 1),
                  "eff90": r(b.eff.quantile(.9), 1), "k0": r(b.kappa0.median(), 3),
                  "util": r((b.kappa0 <= 0.15).mean() * 100, 0), "lnld": r(b.Ln_LD.median(), 2)})

# Ln/LD bins
bins = [0, 0.25, 0.5, 1, 2, 5, np.inf]
lab = ["≤0.25", "0.25–0.5", "0.5–1", "1–2", "2–5", ">5"]
o["bLD"] = pd.cut(o.Ln_LD, bins, labels=lab)
g = o.groupby("bLD", observed=False)
binsLD = {"lab": lab, "n": g.size().tolist(), "eff": [r(x, 1) for x in g.eff.median()],
          "k0": [r(x, 3) for x in g.kappa0.median()],
          "util": [r(x * 100, 0) for x in g.kappa0.apply(lambda s: (s <= 0.15).mean())]}

# per-level medians
niveles = {}
for v in ["dEg", "f_mu", "tau", "Nd_n", "Na_p", "L_n", "L_p"]:
    gg = o.groupby(v)
    niveles[v] = {"x": [float(x) for x in gg.eff.median().index], "eff": [r(x, 2) for x in gg.eff.median()],
                  "k0": [r(x, 3) for x in gg.kappa0.median()],
                  "conv": [r(x * 100, 0) for x in d.groupby(v).ok.mean()]}


def heat(a, b, z):
    p = o.pivot_table(index=a, columns=b, values=z, aggfunc="median")
    return {"x": [float(c) if not isinstance(c, str) else c for c in p.columns],
            "y": [float(i) if not isinstance(i, str) else i for i in p.index],
            "z": [[r(v, 3) for v in row] for row in p.values]}


heats = {f"{a}|{b}": {"eff": heat(a, b, "eff"), "k0": heat(a, b, "kappa0")}
         for a, b in [("tau", "L_n"), ("f_mu", "L_n"), ("Nd_n", "Na_p"), ("tau", "f_mu"), ("material", "dEg"),
                      ("material", "L_n")]}

cols = VARS + ["eff", "kappa0"]
corr = [[r(spearmanr(o[a], o[b])[0], 2) for b in cols] for a in cols]

pts = o[["material", "eff", "kappa0", "Ln_LD", "Eg", "tau", "f_mu", "L_n", "Nd_n", "Na_p"]].copy()
pts["regimen"] = o.regimen.astype(str)
puntos = {c: [r(v, 4) if not isinstance(v, str) else v for v in pts[c]] for c in pts.columns}

top = o[o.eff >= o.eff.quantile(0.9)]
resumen = {"n": len(d), "ok": int(d.ok.sum()), "fail": int((~d.ok).sum()), "util": int((o.kappa0 <= 0.15).sum()),
           "nok": len(o), "rho_k_eff": r(spearmanr(o.kappa0, o.eff)[0], 2),
           "rho_ld_k": r(spearmanr(o.Ln_LD, o.kappa0)[0], 2), "top_thr": r(o.eff.quantile(.9), 1),
           "top_lnld_max": r(top.Ln_LD.max(), 2), "top_k_max": r(top.kappa0.max(), 3),
           "top_mats": top.material.value_counts().to_dict(),
           "util_eff": r(o[o.kappa0 <= 0.15].eff.median(), 1), "prob_eff": r(o[o.kappa0 >= 0.47].eff.median(), 1)}

DATA = json.dumps({"tabla": tabla, "binsLD": binsLD, "niveles": niveles, "heats": heats, "corr": corr,
                   "cols": [NOMBRE.get(c, c) for c in cols], "puntos": puntos, "mats": MATS, "res": resumen},
                  ensure_ascii=False)
html = (RAIZ / "tools/barrido_tpl.html").read_text().replace("/*__DATA__*/null", DATA)
dst.write_text(html)
print(dst, len(html) // 1024, "KB")
print(json.dumps(resumen, ensure_ascii=False))
