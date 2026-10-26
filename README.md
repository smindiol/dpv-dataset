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

Rangos, fuentes y nominales: `dpvgen/simulacion.py` (`FAMILIAS`). Cada familia-etapa se genera en bloques
de 1024 dispositivos de una secuencia Sobol aleatorizada con semilla fija (bloques distintos no se
solapan): 2 bloques en E1 y E3, 3 en E2 y E4, en total 20 480 dispositivos. El bloque b000 lleva ademas
las 2^d esquinas del hipercubo y un 5 % re-simulado con 4x los nodos de la malla base (piso de la Ec. 21).
Mallas base: 500 nodos uniformes en E1 y E3; la malla graduada de ex2 con 4x nodos (720) en E2 y E4
(`FACTOR_MALLA_BASE`, archivos `dpv_<fe>_m4_b<nnn>.h5`). Cada simulacion queda con su estado (`ok`,
`no_convergido`, `sin_cruce`, `nan`, `fallo_newton`, `error`); nada se descarta en silencio.

| Familia-etapa | Sobol ok / total | Esquinas ok | Piso Q95 (pp) |
|---|---|---|---|
| E1A | 2048 / 2048 | 4 / 4 | 0.047 |
| E1B | 2048 / 2048 | 16 / 16 | 0.082 |
| E2A | 3072 / 3072 | 4 / 4 | 0.034 |
| E2B | 3072 / 3072 | 16 / 16 | 0.008 |
| E3A | 2048 / 2048 | 4 / 4 | 0.026 |
| E3B | 1891 / 2048 | 20 / 32 | 0.072 |
| E4A | 3072 / 3072 | 8 / 8 | 0.048 |
| E4B | 3071 / 3072 | 32 / 32 | 0.017 |

Ejecucion: cuadernos autonomos por sesion en `dataset/Notebooks/` (`produccion_s1`..`s4` para E1/E3, `malla4_s5`..`s10`
o `malla4_todo` para E2/E4); cada uno clona este repositorio en un commit fijo y escribe un `.h5` por bloque.
`visualizacion/` tiene un tablero local (`python3 -m http.server --directory visualizacion`) y los scripts que lo
alimentan desde los `.h5`.
