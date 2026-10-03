# Parte 0 · Cómo se optimiza una función de costo

Descenso de gradiente con el ejemplo más simple: ajustar una recta `y = w·x + b` a 50 puntos, calculando el costo y sus derivadas a mano.

Nota: [mikelabs.me/notas/optimizar-una-funcion-de-costo](https://mikelabs.me/notas/optimizar-una-funcion-de-costo/)

```
python costo.py
```

El programa imprime, en orden, los números que aparecen en la nota:

1. El costo y su derivada con un solo parámetro libre.
2. Un paso de descenso calculado desde `(0, 0)`.
3. El descenso completo con tasa 0,1.
4. La comprobación contra mínimos cuadrados (`np.polyfit`).
5. Qué pasa con cuatro tasas de aprendizaje: 0,01, 0,1, 0,4 y 0,5.

La semilla está fija (`default_rng(0)`), así que los resultados son los mismos en cada corrida.
