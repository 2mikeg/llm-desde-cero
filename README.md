# Cómo funciona un LLM, desde cero

El código de la serie que escribo en [mikelabs.me](https://mikelabs.me/desde-cero/). Estoy estudiando a fondo cómo funciona un LLM, y cada parte es lo que voy entendiendo, con el código que escribo para entenderlo.

Cada carpeta corresponde a una nota. El código usa NumPy y nada más: sin PyTorch ni librerías de deep learning, para que cada operación quede a la vista.

| # | Parte | Código | Nota |
|---|-------|--------|------|
| 0 | Optimizar una función de costo | [00-funcion-de-costo](00-funcion-de-costo/) | [Leer](https://mikelabs.me/notas/optimizar-una-funcion-de-costo/) |
| 1 | Una red neuronal con NumPy | Por publicar | |
| 2 | Tokens | Por escribir | |
| 3 | Embeddings | Por escribir | |
| 4 | Atención, calculada a mano | Por escribir | |
| 5 | El bloque transformer | Por escribir | |
| 6 | Entrenar un GPT pequeño | Por escribir | |
| 7 | De modelo base a asistente | Por escribir | |

Las carpetas aparecen cuando se publica la nota correspondiente. El orden y los títulos pueden cambiar según lo que vaya aprendiendo.

## Cómo correrlo

Necesitas Python 3.10 o más reciente y NumPy.

```
pip install -r requirements.txt
python 00-funcion-de-costo/costo.py
```

## Licencia

MIT. Úsalo, cópialo y cámbialo como quieras.
