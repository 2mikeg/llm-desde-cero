"""Parte 0 · Cómo se optimiza una función de costo.

Ajusta una recta y = w·x + b a 50 puntos con descenso de gradiente,
calculando el costo y sus derivadas a mano con NumPy.

Nota: https://mikelabs.me/notas/optimizar-una-funcion-de-costo/
"""

import numpy as np

rng = np.random.default_rng(0)

# Datos: 50 puntos alrededor de la recta y = 2x + 1, con ruido
x = rng.uniform(0, 2, 50)
y = 2 * x + 1 + rng.normal(0, 0.3, 50)


def costo(w, b):
    """Error cuadrático medio de la recta y = w·x + b sobre los datos."""
    return np.mean((w * x + b - y) ** 2)


def gradiente(w, b):
    """Derivadas del costo respecto a w y a b."""
    error = w * x + b - y
    return 2 * np.mean(error * x), 2 * np.mean(error)


def descenso(tasa, pasos, w=0.0, b=0.0, cada=None):
    """Repite la regla de actualización: parámetro -= tasa · derivada."""
    for paso in range(pasos + 1):
        if cada and paso % cada == 0:
            c = costo(w, b)
            print(f"paso {paso:3d}  w={w:.3f}  b={b:.3f}  costo={c:.4f}")
        dw, db = gradiente(w, b)
        w, b = w - tasa * dw, b - tasa * db
    return w, b


if __name__ == "__main__":
    print("Un parámetro a la vez (b fijo en 1)")
    for w in (0, 2, 4):
        print(f"  w={w}  costo={costo(w, 1):.2f}  derivada={gradiente(w, 1)[0]:+.2f}")

    print("\nUn paso a mano desde (0, 0) con tasa 0.1")
    dw, db = gradiente(0, 0)
    w1, b1 = 0 - 0.1 * dw, 0 - 0.1 * db
    print(f"  gradiente=({dw:.2f}, {db:.2f})  ->  w={w1:.3f}  b={b1:.3f}")
    print(f"  costo: {costo(0, 0):.2f} -> {costo(w1, b1):.2f}")

    print("\nDescenso de gradiente, tasa 0.1")
    descenso(tasa=0.1, pasos=200, cada=40)

    print("\nComprobación con mínimos cuadrados")
    w_exacto, b_exacto = np.polyfit(x, y, 1)
    w, b = descenso(tasa=0.1, pasos=1000)
    print(f"  fórmula exacta:        w={w_exacto:.3f}  b={b_exacto:.3f}")
    print(f"  descenso (1000 pasos): w={w:.3f}  b={b:.3f}  costo={costo(w, b):.4f}")

    print("\nCuatro tasas de aprendizaje, 100 pasos cada una")
    for tasa in (0.01, 0.1, 0.4, 0.5):
        w, b = descenso(tasa, 100)
        print(f"  tasa {tasa:<4}  w={w:.3g}  b={b:.3g}  costo={costo(w, b):.3g}")
