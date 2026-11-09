# Innovación del grupo: auditor de sesgo de popularidad.
# Problema: SASRec cubre menos catálogo que el coseno (56,5 % vs 75,6 %) y todos los modelos
# fallan en la cola larga. El auditor mide, para cada lista, cuánto se apoya en lo popular,
# cuán novedosa es y cuán variada, y avisa cuando la lista está sesgada.
from itertools import combinations

import numpy as np

# Umbrales declarados (decisión del grupo, calibrada con los datos): proporción del Top-N que pertenece
# al 20 % de juegos más reseñados. Como referencia, el 73 % de los juegos que los usuarios realmente
# jugaron después (prueba) están en ese 20 %. Una lista con 8 de 10 populares ya supera esa realidad
# (sesgo medio); con 9 o 10 de 10, el sesgo es alto.
UMBRAL_ALTO = 0.9
UMBRAL_MEDIO = 0.8


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0


def auditar(items, rec):
    # items: lista de índices de juegos (en el orden del modelo); rec: Recomendador
    items = [int(i) for i in items]
    if not items:
        return {}
    conteo = rec.conteo[items]
    # Novedad = información propia promedio: -log2(probabilidad de que un usuario haya jugado el juego)
    novedad = float(np.mean(-np.log2((conteo + 1) / (rec.n_usuarios + 1))))
    en_cabeza = float(np.mean([i in rec.cabeza for i in items]))
    generos = [rec.mapa.at[i, "generos"] for i in items]
    pares = list(combinations(range(len(items)), 2))
    diversidad = float(1 - np.mean([jaccard(generos[a], generos[b]) for a, b in pares])) if pares else 0.0
    if en_cabeza >= UMBRAL_ALTO:
        nivel = "alto"
    elif en_cabeza >= UMBRAL_MEDIO:
        nivel = "medio"
    else:
        nivel = "bajo"
    return {
        "proporcion_populares": round(en_cabeza, 3),
        "porcentaje_populares": int(round(100 * en_cabeza)),
        "novedad_bits": round(novedad, 2),
        "diversidad_generos": round(diversidad, 3),
        "generos_distintos": len({rec.mapa.at[i, "genero_principal"] for i in items}),
        "reseñas_mediana": int(np.median(conteo)),
        "sesgo_popularidad": nivel,
    }
