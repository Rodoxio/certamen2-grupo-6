# Certamen II · Grupo 6 — Un recomendador que entiende el orden

Recomendador secuencial de videojuegos de Steam basado en Transformer (SASRec), comparado con las líneas base del curso y publicado como dashboard con una IA generativa que **no recomienda: analiza** lo que el modelo recomendó.

**Integrantes:** Lucas Hermosilla Díaz, Rodolfo Hidalgo Memmel, Ariel Linero Torres
**Docente:** Danilo Gómez Correa · Machine Learning (UDD), 2026-II

**Dashboard:** [certamen2-grupo-6.streamlit.app](https://certamen2-grupo-6.streamlit.app/)

## Rubro y datos

- **Rubro:** videojuegos. **Usuario:** un jugador que busca qué jugar después. **Decisión:** qué juegos mostrarle en "Recomendado para ti".
- **Base:** Steam Video Game Reviews v2 (UCSD; Kang y McAuley, 2018). Tras limpieza, 5-core y una muestra fija de 40.000 usuarios (semilla 2026): **482.019 interacciones, 39.401 usuarios y 6.738 juegos**.

## Resultados (Entrega 1, prueba leave-last-out con ranking del catálogo completo)

| Modelo | HR@10 | NDCG@10 | MRR@20 | Cobertura@10 |
|---|---|---|---|---|
| Popularidad | 0,0552 | 0,0282 | 0,0218 | 0,9 % |
| Reglas de asociación | 0,0572 | 0,0288 | 0,0222 | 0,9 % |
| Ítem–ítem coseno (γ = 0,9) | 0,0861 | 0,0452 | 0,0361 | 75,6 % |
| **SASRec (config C)** | **0,1110** | **0,0580** | **0,0461** | 56,5 % |

## Entrega 2: dashboard y analista con IA

**Regla de oro.** SASRec genera la lista y su orden. El analista (Gemini) recibe un contexto estructurado (historial con atributos, Top-N con probabilidades, pesos de atención, Top-N del coseno y el auditor de sesgo) y solo habla de esos juegos. Si una pregunta exige otra lista, la app vuelve a consultar a SASRec con un filtro. Cada respuesta pasa por una verificación automática: no puede nombrar juegos fuera del contexto y cada `#k` debe corresponder a la posición real.

**Innovación: auditor de sesgo de popularidad.** Mide en cada lista el % de juegos populares, la novedad y la diversidad, y el analista lo advierte. Sobre los 39.401 usuarios, el 93,8 % del Top-10 de SASRec está en el 20 % de juegos más reseñados, contra el 73,3 % de lo que los usuarios realmente jugaron después: el modelo más preciso también es el más sesgado a lo popular.

## Estructura

```
app.py                         dashboard en Streamlit
src/
  recomendador.py              SASRec (carga sasrec.pt en CPU), popularidad e ítem–ítem coseno
  analista.py                  llamadas a Gemini, contexto, tokens, latencia y verificación de honestidad
  analista_prompt.md           instrucciones de sistema versionadas del analista y del clasificador
  auditor.py                   innovación: auditor de sesgo de popularidad
  flujo.py                     flujo compartido por la app y el notebook 2
Certamen2_Entrega1_G6.ipynb    notebook 1: datos, líneas base, SASRec y comparación
Certamen2_Entrega2_G6.ipynb    notebook 2: diseño del analista, pruebas de honestidad, innovación y costo
Informe_Entrega1_Grupo6_FINAL.pdf  informe de la Entrega 1
data/
  secuencias.parquet           secuencia ordenada de juegos por usuario (usuarios anonimizados)
  mapa_items.csv               índice del modelo → juego, géneros, precio y popularidad
  metricas.csv                 métricas de prueba de los cuatro modelos
  auditoria_global.csv         auditor de sesgo sobre todos los usuarios (notebook 2)
modelos/
  sasrec.pt                    pesos y configuración de SASRec
requirements.txt               dependencias (PyTorch CPU)
.streamlit/config.toml         tema de la app con los colores de Steam
```

## Ejecutar

- **Local:** `pip install -r requirements.txt`, crear `.streamlit/secrets.toml` con `GEMINI_API_KEY = "..."` y correr `streamlit run app.py`.
- **Streamlit Cloud:** la clave se configura en *Settings → Secrets*. Nunca se sube al repositorio (está en `.gitignore`).
- **Notebooks:** abrir en Colab con GPU; el notebook 2 lee la clave desde los secretos de Colab.

## Declaración de uso de IA

Para programar usamos Claude (Anthropic). Dentro del producto, el analista usa Gemini (Google).
