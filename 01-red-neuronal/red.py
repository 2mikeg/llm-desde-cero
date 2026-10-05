"""Parte 1 · Una red neuronal con NumPy.

Una red 2 -> 4 -> 1 (tanh en la capa oculta) que separa dos medias lunas.
El forward y backpropagation están escritos a mano con NumPy, y las
derivadas se comprueban contra diferencias finitas.

Nota: https://mikelabs.me/notas/red-neuronal-con-numpy/
"""

import numpy as np

rng = np.random.default_rng(0)

# Datos: 200 puntos en dos medias lunas, 100 de cada clase.
# La clase 0 son los puntos blancos de las figuras y la 1, los azules
t = rng.uniform(0, np.pi, 100)
luna0 = np.c_[np.cos(t), np.sin(t)]
luna1 = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
X = np.r_[luna0, luna1] + rng.normal(0, 0.1, (200, 2))
y = np.r_[np.zeros(100), np.ones(100)].reshape(-1, 1)

# Parámetros: pesos aleatorios y sesgos en cero. Son 17 en total
W1 = rng.normal(0, 1, (2, 4))    # 2 entradas -> 4 neuronas
b1 = np.zeros(4)
W2 = rng.normal(0, 0.5, (4, 1))  # 4 neuronas -> 1 salida
b2 = np.zeros(1)
params = [W1, b1, W2, b2]
inicio = [p.copy() for p in params]


def forward():
    """De los datos a la predicción. Devuelve también la capa oculta."""
    h = np.tanh(X @ W1 + b1)  # (200, 4)
    return h, h @ W2 + b2     # (200, 1)


def costo(pred):
    """Error cuadrático medio, el mismo de la parte 0."""
    return np.mean((pred - y) ** 2)


def acierto(pred):
    """Fracción de puntos bien clasificados: clase 1 si pred > 0.5."""
    return np.mean((pred > 0.5) == y)


def backward(h, pred):
    """Backpropagation: las derivadas del costo, de la salida hacia atrás."""
    d2 = 2 * (pred - y) / len(y)     # costo -> predicción
    dW2, db2 = h.T @ d2, d2.sum(0)
    d1 = (d2 @ W2.T) * (1 - h**2)    # predicción -> h -> tanh
    dW1, db1 = X.T @ d1, d1.sum(0)
    return dW1, db1, dW2, db2


def entrenar(tasa, pasos, cada=400):
    """Descenso de gradiente: parámetro -= tasa · derivada, `pasos` veces.
    Devuelve el costo y los aciertos de cada paso."""
    historia = []
    for paso in range(pasos + 1):
        h, pred = forward()
        historia.append((costo(pred), int(np.sum((pred > 0.5) == y))))
        if cada and paso % cada == 0:
            print(f"paso {paso:4d}  costo={costo(pred):.4f}  "
                  f"acierto={acierto(pred):.1%}")
        if paso < pasos:
            # `p -= ...` modifica el arreglo en su sitio: cambia W1, b1, W2, b2
            grads = backward(h, pred)
            for p, g in zip(params, grads):
                p -= tasa * g
    return historia


def reiniciar():
    """Devuelve los parámetros a sus valores iniciales."""
    for p, p0 in zip(params, inicio):
        p[:] = p0


def diferencias_finitas(eps=1e-6):
    """Derivada numérica de cada parámetro: lo muevo eps hacia arriba y
    hacia abajo y miro cuánto cambia el costo."""
    grads = []
    for p in params:
        g = np.zeros_like(p)
        for i in np.ndindex(p.shape):
            original = p[i]
            p[i] = original + eps
            arriba = costo(forward()[1])
            p[i] = original - eps
            abajo = costo(forward()[1])
            p[i] = original
            g[i] = (arriba - abajo) / (2 * eps)
        grads.append(g)
    return grads


def entrenar_sin_tanh(tasa, pasos):
    """La misma red, con los mismos valores iniciales, pero sin la tanh:
    h = X·W1 + b1. Sirve para ver que dos capas sin la tanh no hacen más
    que una recta."""
    V1, c1, V2, c2 = [p.copy() for p in inicio]
    for _ in range(pasos):
        h = X @ V1 + c1
        d2 = 2 * (h @ V2 + c2 - y) / len(y)
        d1 = d2 @ V2.T               # sin tanh no hay (1 - h**2)
        dV1, dc1, dV2, dc2 = X.T @ d1, d1.sum(0), h.T @ d2, d2.sum(0)
        V1 -= tasa * dV1
        c1 -= tasa * dc1
        V2 -= tasa * dV2
        c2 -= tasa * dc2
    return (X @ V1 + c1) @ V2 + c2


def mejor_recta():
    """Prueba todas las posiciones de los dos deslizadores de la primera
    figura de la nota. La frontera es x2 = inclinacion · x1 + altura, y
    la clase 1 es lo que queda debajo."""
    mejor = (0, 0.0, 0.0)
    for inclinacion in np.arange(-200, 201) / 100:
        for altura in np.arange(-100, 151) / 100:
            debajo = X[:, 1:] < inclinacion * X[:, :1] + altura
            bien = int(np.sum(debajo == y))
            if bien > mejor[0]:
                mejor = (bien, inclinacion, altura)
    return mejor


def mejor_recta_posible():
    """El tope para cualquier recta, no solo las de los deslizadores.
    Toda forma de partir los puntos con una recta se consigue con una que
    pasa por dos de ellos, así que basta probar los 19 900 pares. Los dos
    puntos que quedan sobre la recta se cuentan como aciertos: moviéndola
    apenas, cada uno cae del lado que le conviene."""
    mejor = 0
    azul = y.ravel() == 1
    for i in range(len(X) - 1):
        for j in range(i + 1, len(X)):
            dx, dy = X[j] - X[i]
            lado = (X[:, 0] - X[i, 0]) * dy - (X[:, 1] - X[i, 1]) * dx > 0
            bien = int(np.sum(np.delete(lado == azul, [i, j])))
            mejor = max(mejor, bien + 2, 198 - bien + 2)
    return mejor


def otros_inicios(cuantos=40, tasa=0.2, pasos=2000):
    """La misma red y los mismos datos con otros pesos iniciales, uno por
    semilla (de 1 a `cuantos`). Devuelve los aciertos finales de cada una."""
    aciertos = []
    for semilla in range(1, cuantos + 1):
        r = np.random.default_rng(semilla)
        V1, c1 = r.normal(0, 1, (2, 4)), np.zeros(4)
        V2, c2 = r.normal(0, 0.5, (4, 1)), np.zeros(1)
        for _ in range(pasos):
            h = np.tanh(X @ V1 + c1)
            d2 = 2 * (h @ V2 + c2 - y) / len(y)
            d1 = (d2 @ V2.T) * (1 - h**2)
            dV1, dc1, dV2, dc2 = X.T @ d1, d1.sum(0), h.T @ d2, d2.sum(0)
            V1 -= tasa * dV1
            c1 -= tasa * dc1
            V2 -= tasa * dV2
            c2 -= tasa * dc2
        pred = np.tanh(X @ V1 + c1) @ V2 + c2
        aciertos.append(int(np.sum((pred > 0.5) == y)))
    return aciertos


if __name__ == "__main__":
    print("Una recta (mínimos cuadrados) sobre las dos lunas")
    A = np.c_[X, np.ones(len(X))]
    w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0].ravel()
    recta = A @ [[w1], [w2], [b]]
    print(f"  w1={w1:.3f}  w2={w2:.3f}  b={b:.3f}")
    print(f"  costo={costo(recta):.4f}  acierto={acierto(recta):.1%}  "
          f"({int(np.sum((recta > 0.5) == y))} de {len(y)})")
    # La frontera es donde la recta vale 0.5: x2 = inclinación · x1 + altura
    print(f"  frontera: inclinación={-w1 / w2:.2f}  altura={(0.5 - b) / w2:.2f}")
    bien, inclinacion, altura = mejor_recta()
    print(f"  la mejor recta de los deslizadores: {bien} de {len(y)}  "
          f"(inclinación={inclinacion:.2f}  altura={altura:.2f})")
    print(f"  la mejor de todas las rectas posibles: "
          f"{mejor_recta_posible()} de {len(y)}")

    print("\nLa red 2 -> 4 -> 1 antes de entrenar")
    h, pred = forward()
    print(f"  parámetros={sum(p.size for p in params)}")
    print(f"  costo={costo(pred):.4f}  acierto={acierto(pred):.1%}")

    print("\nLa misma red sin la tanh, 2000 pasos con tasa 0.2")
    lineal = entrenar_sin_tanh(tasa=0.2, pasos=2000)
    print(f"  costo={costo(lineal):.4f}  acierto={acierto(lineal):.1%}  "
          f"({int(np.sum((lineal > 0.5) == y))} de {len(y)})")

    print("\nComprobación del gradiente: backprop contra diferencias finitas")
    analitico = backward(h, pred)
    numerico = diferencias_finitas()
    # W1[0, 1]: el peso que va de la primera entrada a la segunda neurona
    print(f"  dJ/dW1[0,1]  backprop={analitico[0][0, 1]:+.6f}  "
          f"diferencias={numerico[0][0, 1]:+.6f}")
    dif = max(np.abs(a - n).max() for a, n in zip(analitico, numerico))
    print(f"  mayor diferencia en los 17 parámetros: {dif:.1e}")
    print(f"  forward usados por diferencias finitas: {2 * 17}")

    print("\nEntrenamiento, tasa 0.2")
    historia = entrenar(tasa=0.2, pasos=2000)
    ultimo = max(i for i, (_, bien) in enumerate(historia) if bien < 200)
    print(f"  200 de 200 desde el paso {ultimo + 1}")

    # Con 0.3 el costo salta a mitad de camino. Dentro del salto los
    # decimales pueden cambiar de una máquina a otra: ahí cualquier
    # diferencia de redondeo se amplifica
    print("\nLa misma red con tasa 0.3")
    reiniciar()
    historia = entrenar(tasa=0.3, pasos=2000, cada=None)
    costos = [c for c, _ in historia]
    sube = next(i for i in range(100, 2000) if costos[i + 1] > costos[i])
    pico = max(range(sube, 2001), key=lambda i: costos[i])
    print(f"  baja hasta el paso {sube}: costo={costos[sube]:.3f}")
    print(f"  salta hasta el paso {pico}: costo={costos[pico]:.2f}  "
          f"aciertos mínimos={min(bien for _, bien in historia[sube:])}")
    print(f"  paso 2000: costo={costos[2000]:.4f}  "
          f"acierto={historia[2000][1] / 200:.1%}")

    print("\nOtros 40 valores iniciales, tasa 0.2, 2000 pasos")
    aciertos = otros_inicios()
    for n in sorted(set(aciertos), reverse=True):
        print(f"  {n} de 200: {aciertos.count(n)} de {len(aciertos)}")
