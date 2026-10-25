# dpv-dataset

Generador del dataset sintetico de referencia para un benchmark de configuraciones PINN sobre el sistema
acoplado de Poisson y deriva-difusion en celdas solares de pelicula delgada. La referencia numerica es
[∂PV (deltapv)](https://github.com/romanodev/deltapv), fijado al commit `f874e24`.

| Familia | Dispositivo | xi en la etapa A | xi en la etapa B |
|---|---|---|---|
| E1 | Homounion p-n (`ex1`) | Nd, Na | + L_n, L_p |
| E2 | CdS/CdTe (`ex2`) | Nd_CdS, Na_CdTe | + L_CdS, L_CdTe |
| E3 | Perovskita ETM/abs/HTM (`psc.py`) | Nd_ETM, Na_HTM | + L_abs, L_ETL, L_HTL |
| E4 | CdS/CdTe + capa LDS (transformacion espectral) | Nd_CdS, Na_CdTe, eta_PL | + L_CdS, L_CdTe |

Rangos, fuentes y nominales: `dpvgen/simulacion.py` (`FAMILIAS`). Por familia-etapa se generan 12 500
dispositivos Sobol aleatorizados (semilla fija), las 2^d esquinas del hipercubo y un 5 % re-simulado con
4x nodos. Cada simulacion queda con su estado (`ok`, `no_convergido`, `sin_cruce`, `nan`,
`fallo_newton`, `error`); nada se descarta en silencio.

Ejecucion: `dataset/Notebooks/generar_dataset_dpv.ipynb` (modo TEST o PRODUCCION; ver la primera celda).
