# Parte 1 · Una red neuronal con NumPy

Forward y backpropagation escritos a mano: una red de 2 entradas, 4 neuronas ocultas con `tanh` y 1 salida, que aprende a separar 200 puntos en forma de dos medias lunas. Tiene 17 parámetros y se entrena con el mismo descenso de gradiente de la parte 0.

- **Nota:** en curso. Va a estar en `mikelabs.me/notas/red-neuronal-con-numpy/`.

```
python red.py
```

El programa imprime, en orden, los números que aparecen en la nota:

1. Cuántos puntos acierta una recta: la de mínimos cuadrados, la mejor que se consigue con los dos deslizadores de la primera figura y la mejor de todas las rectas posibles (prueba las 19 900 que pasan por dos puntos, que cubren todas las formas de partir los datos con una recta).
2. El costo de la red antes de entrenar.
3. La misma red sin la `tanh`, que termina igual que la recta.
4. La comprobación del gradiente: backpropagation contra diferencias finitas, en un peso y en los 17 parámetros.
5. El entrenamiento con tasa 0,2 durante 2000 pasos, y desde qué paso acierta los 200 puntos.
6. La misma red con tasa 0,3, donde el costo salta a mitad de camino.
7. La misma red con otros 40 valores iniciales (semillas 1 a 40): cuántas llegan a 200 aciertos y cuántas se quedan cerca.

La semilla está fija (`default_rng(0)`), así que los resultados son los mismos en cada corrida. La excepción son los números del salto con tasa 0,3: ahí el entrenamiento es inestable y una diferencia en el último decimal se amplifica, así que el paso exacto y la altura del salto cambian un poco de una máquina a otra. Antes y después del salto los números coinciden.

La segunda figura de la nota entrena esta misma red en el navegador, con los mismos puntos y los mismos valores iniciales. Con tasa 0,2 muestra en cada paso el costo que imprime este programa.
