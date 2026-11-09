# Dashboard del Certamen II · Grupo 6
# Un recomendador que entiende el orden: SASRec recomienda, la IA (Gemini) solo analiza.
import hashlib
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

from src.analista import Analista, cargar_prompts, verificar_honestidad
from src.flujo import calcular, preguntar
from src.recomendador import Recomendador

RAIZ = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Recomendador de juegos · Grupo 6", page_icon="🎮", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.cabecera {background: linear-gradient(90deg, #171a21 0%, #1b2838 55%, #2a475e 100%);
           border-bottom: 3px solid #66c0f4; padding: 18px 26px; border-radius: 6px; margin-bottom: 14px;}
.cabecera h1 {color: #ffffff; font-size: 1.9rem; margin: 0; letter-spacing: .5px;}
.cabecera p {color: #8f98a0; margin: 4px 0 0 0;}
.etiqueta {display:inline-block; background:#2a475e; color:#66c0f4; padding:2px 10px; border-radius:3px;
           font-size:.78rem; letter-spacing:1px; text-transform:uppercase;}
.verde {background:#5c7e10; color:#d2efa9;} .rojo {background:#7a2b2b; color:#f3c4c4;}
.amarillo {background:#7a6a1e; color:#f7ecb5;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Recursos con caché (se cargan una sola vez por servidor)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando SASRec y las líneas base...")
def cargar_recomendador():
    return Recomendador(RAIZ)


@st.cache_resource
def cargar_analista():
    prompts = cargar_prompts(os.path.join(RAIZ, "src", "analista_prompt.md"))
    try:
        clave = st.secrets.get("GEMINI_API_KEY", None)
        modelo = st.secrets.get("GEMINI_MODEL", None)
    except Exception:
        clave, modelo = os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_MODEL")
    if not clave:
        return None, prompts
    modelos = [modelo] if modelo else None
    return Analista(clave, prompts, modelos), prompts


rec = cargar_recomendador()
analista, prompts = cargar_analista()
ejemplos = rec.ejemplos()

# Etiquetas únicas para el buscador de juegos (hay nombres repetidos en Steam)
repetidos = rec.mapa["nombre"].duplicated(keep=False)
etiqueta_de = {it: (f"{f.nombre} [{f.product_id}]" if repetidos[it] else f.nombre) for it, f in rec.mapa.iterrows()}
item_de = {v: k for k, v in etiqueta_de.items()}
opciones = [etiqueta_de[i] for i in sorted(etiqueta_de, key=lambda i: -rec.conteo[i])]
conteo_generos = rec.mapa["generos"].explode().value_counts()
generos_validos = sorted(conteo_generos[conteo_generos >= 30].index)

for clave, valor in [("chat", []), ("consumo", []), ("analisis", {}), ("historial_sel", [])]:
    st.session_state.setdefault(clave, valor)


def registrar(r, tipo):
    st.session_state.consumo.append({"tipo": tipo, "modelo": r["modelo"], "tokens_entrada": r["tokens_entrada"],
                                     "tokens_salida": r["tokens_salida"], "latencia_s": r["latencia_s"]})


def insignia(verif):
    if verif["ok"]:
        return '<span class="etiqueta verde">✔ verificación: solo juegos y posiciones del modelo</span>'
    detalle = "; ".join(verif["fuera_de_lista"][:3] + verif["rangos_incorrectos"][:3])
    return f'<span class="etiqueta rojo">✖ verificación: {detalle}</span>'


def tabla(df, col):
    vista = df[["#", "juego", "género", "precio", "reseñas", col]].copy()
    return st.dataframe(vista, hide_index=True, width="stretch",
                        column_config={"precio": st.column_config.NumberColumn("precio US$", format="%.2f"),
                                       col: st.column_config.NumberColumn(col, format="%.2f")})


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    n = st.slider("Tamaño de la lista (Top-N)", 5, 20, 10)
    st.markdown("### 🤖 Analista")
    if analista:
        st.success("Gemini conectado")
    else:
        st.warning("Sin clave: agrega GEMINI_API_KEY en los secrets de Streamlit para activar el analista.")
    consumo = pd.DataFrame(st.session_state.consumo)
    if len(consumo):
        st.metric("Llamadas al LLM en esta sesión", len(consumo))
        c1, c2 = st.columns(2)
        c1.metric("Tokens entrada", f"{consumo.tokens_entrada.sum():,}".replace(",", "."))
        c2.metric("Tokens salida", f"{consumo.tokens_salida.sum():,}".replace(",", "."))
        st.metric("Latencia media", f"{consumo.latencia_s.mean():.1f} s")
    if st.button("Limpiar chat"):
        st.session_state.chat = []
    st.caption("Modelo: SASRec (config C, d=128, 2 bloques). Inferencia en CPU; nada se entrena en la app.")

st.markdown('<div class="cabecera"><h1>🎮 Un recomendador que entiende el orden</h1>'
            '<p>SASRec lee tu historial como una secuencia y predice tu próximo juego. '
            'La IA no recomienda: analiza lo que el modelo recomendó.</p></div>', unsafe_allow_html=True)

pestanas = st.tabs(["🕹️ Simulador", "📚 Catálogo", "📊 Métricas", "ℹ️ Cómo funciona"])

# ---------------------------------------------------------------------------
# Simulador
# ---------------------------------------------------------------------------
with pestanas[0]:
    presets = {"Jugador real · historial corto (5 juegos)": "corto",
               "Jugador real · historial medio (13 juegos)": "medio",
               "Jugador real · historial largo (42 juegos)": "largo",
               "Arranque en frío (sin juegos)": "frio", "Armar mi propio historial": "propio"}
    elegido = st.selectbox("Historial de partida", list(presets), index=2)
    if st.session_state.get("preset") != elegido:          # al cambiar el preset se carga su historial
        st.session_state.preset = elegido
        clave = presets[elegido]
        st.session_state.historial_sel = ([etiqueta_de[i] for i in ejemplos[clave]["historial"]]
                                          if clave in ejemplos else [])
        st.session_state.chat = []
    seleccion = st.multiselect("Juegos en el orden en que los jugó (el último es el más reciente)",
                               opciones, key="historial_sel", placeholder="Busca un juego...")
    historial = [item_de[s] for s in seleccion]

    res = calcular(rec, historial, n)
    clave_preset = presets[elegido]
    if clave_preset in ejemplos and historial == ejemplos[clave_preset]["historial"]:
        real = ejemplos[clave_preset]["siguiente_real"]
        lista = list(res["top_s"]["item"])
        donde = f"SASRec lo puso en el **#{lista.index(real) + 1}**" if real in lista else f"no aparece en el Top-{n} de SASRec"
        st.info(f"🎯 Juego que este jugador reseñó después (oculto al modelo): **{rec.nombre(real)}** — {donde}.")

    if not historial:
        st.warning("**Arranque en frío:** sin historial, SASRec no tiene secuencia que leer. "
                   "Se muestra la lista de popularidad y el analista lo advierte.")
    elif len(historial) <= 2:
        st.warning("**Historial muy corto:** SASRec recomienda con poca información.")

    col1, col2 = st.columns(2)
    with col1:
        if res["top_s"] is not None:
            st.markdown('<span class="etiqueta">Transformer · SASRec</span>', unsafe_allow_html=True)
            tabla(res["top_s"], "prob. %")
        else:
            st.markdown('<span class="etiqueta">Popularidad (arranque en frío)</span>', unsafe_allow_html=True)
            tabla(res["top_p"], "reseñas en entrenamiento")
    with col2:
        st.markdown('<span class="etiqueta">Línea base · ítem–ítem coseno (decaimiento 0,9)</span>' if historial
                    else '<span class="etiqueta">Línea base · popularidad</span>', unsafe_allow_html=True)
        tabla(res["top_c"], "puntaje" if historial else "reseñas en entrenamiento")

    col3, col4 = st.columns([1.1, 1])
    with col3:
        st.markdown("#### 🔍 Transparencia: ¿en qué se fijó el modelo?")
        if res["atencion"] is not None:
            at = res["atencion"].tail(15)
            grafico = alt.Chart(at).mark_bar(color="#66c0f4").encode(
                x=alt.X("peso:Q", title="peso de atención"),
                y=alt.Y("juego:N", sort=list(at["juego"]), title=None),
                tooltip=["posición", "juego", alt.Tooltip("peso:Q", format=".3f")]).properties(height=28 * len(at) + 30)
            st.altair_chart(grafico, width="stretch")
            st.caption("Pesos de la última posición (la que genera la recomendación), último bloque, "
                       "promediando las 2 cabezas. Abajo está el juego más reciente. Atención no es causalidad.")
        else:
            st.caption("Sin historial no hay atención que mostrar.")
    with col4:
        st.markdown("#### ⚖️ Auditor de sesgo de popularidad")
        aud = pd.DataFrame(res["auditoria"]).T.drop(columns="proporcion_populares").rename(columns={
            "porcentaje_populares": "% populares", "novedad_bits": "novedad (bits)",
            "diversidad_generos": "diversidad", "generos_distintos": "géneros", "reseñas_mediana": "reseñas (mediana)",
            "sesgo_popularidad": "sesgo"})
        st.dataframe(aud, width="stretch")
        nivel = list(res["auditoria"].values())[0]["sesgo_popularidad"]
        color = {"alto": "rojo", "medio": "amarillo", "bajo": "verde"}[nivel]
        st.markdown(f'<span class="etiqueta {color}">sesgo de la lista principal: {nivel}</span>', unsafe_allow_html=True)
        st.caption("% populares = parte de la lista en el 20 % de juegos más reseñados. Novedad = información "
                   "propia promedio (más alto = menos conocido). Diversidad = 1 − parecido promedio de géneros.")

    # ----------------------- Analista con IA -----------------------
    st.markdown("### 🤖 Analista con IA")
    if analista is None:
        st.info("El analista está desactivado porque no hay clave configurada.")
    else:
        huella = hashlib.md5(json.dumps(res["contexto"], sort_keys=True).encode()).hexdigest()
        if huella not in st.session_state.analisis:          # análisis inicial automático (con caché)
            with st.spinner("El analista está leyendo la recomendación..."):
                try:
                    r = analista.analisis_inicial(res["contexto"])
                    registrar(r, "análisis inicial")
                    r["verificacion"] = verificar_honestidad(r["texto"], res["contexto"], rec.mapa["nombre"])
                    st.session_state.analisis[huella] = r
                except Exception as e:
                    st.error(f"No se pudo consultar al LLM: {e}")
        r = st.session_state.analisis.get(huella)
        if r:
            st.markdown(r["texto"])
            st.markdown(insignia(r["verificacion"]), unsafe_allow_html=True)
            st.caption(f"{r['modelo']} · {r['tokens_entrada']} tokens de entrada · {r['tokens_salida']} de salida · "
                       f"{r['latencia_s']} s")

        for msg in st.session_state.chat:
            with st.chat_message(msg["rol"]):
                st.markdown(msg["texto"])
                if msg.get("tabla") is not None:
                    st.dataframe(msg["tabla"], hide_index=True, width="stretch")
                if msg.get("pie"):
                    st.markdown(msg["pie"], unsafe_allow_html=True)

        pregunta = st.chat_input("Pregúntale al analista (ej.: ¿y sin juegos de acción?, ¿por qué el #1?)")
        if pregunta:
            st.session_state.chat.append({"rol": "user", "texto": pregunta})
            try:
                previo = [(m1["texto"], m2["texto"]) for m1, m2 in
                          zip(st.session_state.chat[:-1:2], st.session_state.chat[1::2])]
                salida = preguntar(analista, rec, historial, pregunta, n, generos_validos, previo)
                for tipo, llamada in salida["llamadas"]:
                    registrar(llamada, tipo)
                r, nuevo = salida["respuesta"], salida["resultado"]
                tabla_nueva = None
                if salida["reconsultado"]:
                    tabla_nueva = (nuevo["top_s"] if nuevo["top_s"] is not None else nuevo["top_p"])[
                        ["#", "juego", "género", "precio"]]
                pie = (insignia(r["verificacion"]) + f"<br><small>{r['modelo']} · {r['tokens_entrada']}+"
                       f"{r['tokens_salida']} tokens · {r['latencia_s']} s"
                       + (" · 🔄 lista reconsultada al modelo" if salida["reconsultado"] else "") + "</small>")
                st.session_state.chat.append({"rol": "assistant", "texto": r["texto"], "tabla": tabla_nueva, "pie": pie})
            except Exception as e:
                st.session_state.chat.append({"rol": "assistant", "texto": f"No se pudo consultar al LLM: {e}"})
            st.rerun()

# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------
with pestanas[1]:
    st.markdown("#### Los 15 juegos más reseñados (entrenamiento)")
    top = rec.popularidad([], 15)
    st.altair_chart(alt.Chart(top).mark_bar(color="#66c0f4").encode(
        x=alt.X("reseñas:Q"), y=alt.Y("juego:N", sort="-x", title=None),
        tooltip=["juego", "género", "reseñas"]).properties(height=420), width="stretch")
    st.markdown("#### Explorar el catálogo")
    c1, c2, c3 = st.columns([2, 2, 1])
    texto = c1.text_input("Buscar por nombre")
    generos_sel = c2.multiselect("Género", generos_validos)
    precio_max = c3.number_input("Precio máximo (US$)", 0.0, 100.0, 100.0, step=5.0)
    cat = rec.mapa.assign(reseñas=rec.conteo[rec.mapa.index]).reset_index()
    if texto:
        cat = cat[cat["nombre"].str.contains(texto, case=False, regex=False)]
    if generos_sel:
        cat = cat[cat["generos"].apply(lambda g: bool(set(g) & set(generos_sel)))]
    cat = cat[cat["precio"].fillna(0) <= precio_max].sort_values("reseñas", ascending=False)
    st.dataframe(cat[["nombre", "genero_principal", "precio", "release_date", "reseñas"]].head(200),
                 hide_index=True, width="stretch")
    st.caption(f"{len(cat):,} juegos que cumplen los filtros (se muestran hasta 200).".replace(",", "."))

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
with pestanas[2]:
    st.markdown("#### Resultados en prueba (Entrega 1)")
    st.dataframe(rec.metricas.set_index("modelo").style.format("{:.4f}").highlight_max(axis=0, color="#2a475e"),
                 width="stretch")
    st.caption("Leave-last-out: el último juego de cada usuario se oculta y cada modelo ordena los 6.738 juegos. "
               "39.401 usuarios. SASRec supera al coseno en 29 % en HR@10, pero cubre menos catálogo (56,5 % vs 75,6 %).")
    ruta_aud = os.path.join(RAIZ, "data", "auditoria_global.csv")
    if os.path.exists(ruta_aud):
        st.markdown("#### Innovación: auditor de sesgo sobre todos los usuarios de prueba")
        st.dataframe(pd.read_csv(ruta_aud).set_index("modelo"), width="stretch")
        st.caption("Promedio del Top-10 de cada modelo sobre los 39.401 usuarios (notebook 2).")

# ---------------------------------------------------------------------------
# Cómo funciona
# ---------------------------------------------------------------------------
with pestanas[3]:
    st.markdown("""
#### Regla de oro
**El Transformer recomienda; la IA analiza.** La lista y su orden salen de SASRec. El analista (Gemini) recibe un
contexto estructurado (historial con atributos, Top-N con probabilidades, pesos de atención, Top-N del coseno y el
auditor de sesgo) y solo puede hablar de esos juegos. Si una pregunta exige otra lista (por ejemplo, *sin juegos de
acción* o *¿y si después juego X?*), la app **vuelve a consultar al modelo** con ese filtro y el analista comenta la
nueva lista. Cada respuesta pasa por una **verificación automática**: que no nombre juegos fuera del contexto y que
cada `#k` corresponda a la posición real.

#### Flujo
`historial en orden → SASRec (CPU) → Top-N + atención → auditor de sesgo → contexto JSON → Gemini → análisis + verificación`

#### Datos y privacidad
Reseñas de Steam (UCSD, Kang y McAuley, 2018). Los usuarios son números; ni la app, ni el contexto del LLM ni el
repositorio contienen nombres de usuario. La clave del LLM vive en `st.secrets`, nunca en GitHub.

#### Límites
Predecir el siguiente juego no explica por qué la persona lo eligió. La fecha de las reseñas es por día, la reseña
es una señal imperfecta de compra y todos los modelos fallan en los juegos de cola larga.

#### Declaración de uso de IA
Para programar usamos Claude (Anthropic). Dentro del producto, el analista usa Gemini (Google).
""")
    st.caption(f"Instrucciones del analista: src/analista_prompt.md · modelos LLM: {', '.join(analista.modelos) if analista else '—'}")
