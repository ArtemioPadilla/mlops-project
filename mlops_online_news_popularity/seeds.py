import os
import random

import numpy as np

# Semilla global
SEED = int(os.getenv("RP_SEED", "42"))


def set_global_seed(seed: int = SEED) -> int:
    """
    Configura la semilla de todas las librerías que usamos
    para que el entrenamiento sea lo más determinista posible.
    Devuelve la semilla usada (por si quieres loguearla).
    """
    # Python "puro"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy - set both legacy and new API for maximum compatibility
    np.random.seed(seed)  # Legacy API (still widely used)
    # Note: np.random.default_rng(seed) creates a new generator but doesn't set global state

    # Reducir no-determinismo de BLAS (OpenBLAS, MKL, etc.)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    return seed
