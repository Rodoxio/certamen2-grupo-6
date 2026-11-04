# Certamen II · Grupo 6 — Un recomendador que entiende el orden

Recomendador secuencial de videojuegos de Steam basado en Transformer (SASRec), comparado con las líneas base del curso.

**Integrantes:** Lucas Hermosilla Díaz, Rodolfo Hidalgo Memmel, Ariel Linero Torres
**Docente:** Danilo Gómez Correa · Machine Learning (UDD), 2026-II

## Rubro y datos

- **Rubro:** videojuegos. **Usuario:** un jugador que busca qué jugar después. **Decisión:** qué juegos mostrarle en "Recomendado para ti".
- **Base:** Steam Video Game Reviews v2 (UCSD; Kang y McAuley, 2018): reseñas con usuario, juego y fecha, más metadatos de juegos (géneros, precio).
- Tras limpieza, 5-core y una muestra fija de 40.000 usuarios (semilla 2026): **482.019 interacciones, 39.401 usuarios y 6.738 juegos**.

## Resultados (Entrega 1, prueba leave-last-out con ranking del catálogo completo)

| Modelo | HR@10 | NDCG@10 | MRR@20 | Cobertura@10 |
|---|---|---|---|---|
| Popularidad | 0,0552 | 0,0282 | 0,0218 | 0,9 % |
| Reglas de asociación | 0,0572 | 0,0288 | 0,0222 | 0,9 % |
| Ítem–ítem coseno (γ = 0,9) | 0,0861 | 0,0452 | 0,0361 | 75,6 % |
| **SASRec (config C)** | **0,1110** | **0,0580** | **0,0461** | 56,5 % |

SASRec supera a la mejor línea base en un 29 % en HR@10.

## Estructura

```
Certamen2_Avance1_Grupo6.ipynb   notebook de la Entrega 1, ejecutado con salidas y semilla fija
data/
  secuencias.parquet   secuencia ordenada de juegos por usuario (usuarios anonimizados con un número)
  mapa_items.csv       índice del modelo → id de Steam, nombre, géneros, precio y popularidad
  metricas.csv         métricas de prueba de los cuatro modelos
modelos/
  sasrec.pt            pesos y configuración de SASRec (se carga en CPU)
```

## Reproducir

1. Abrir `Certamen2_Avance1_Grupo6.ipynb` en Google Colab con GPU (T4).
2. Ejecutar todo: descarga los datos desde la UCSD a Google Drive (`Certamen2_Steam`).

Los datos originales (1,3 GB) no se suben al repositorio por tamaño y porque contienen nombres de usuario reales.

## Cargar el modelo

```python
import torch
guardado = torch.load("modelos/sasrec.pt", map_location="cpu")
# guardado["config"], guardado["n_items"], guardado["largo"], guardado["estado"]
```

La clase `SASRec` está definida en la sección 7 del notebook.
