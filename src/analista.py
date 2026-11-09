# Analista con IA generativa (Gemini). El Transformer recomienda; el analista solo analiza.
# Incluye: contexto estructurado, llamadas con registro de tokens y latencia, clasificador para
# reconsultar al modelo y una verificación automática de honestidad de cada respuesta.
import difflib
import json
import re
import time

# Se prueban en orden; así la app sigue funcionando aunque Google renombre o retire un modelo
# Flash-Lite primero: en las pruebas fue el más rápido y el de cuota gratuita más holgada
MODELOS_POR_DEFECTO = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash"]


def cargar_prompts(ruta):
    # Lee cada sección "## Sistema: nombre" del archivo versionado
    texto = open(ruta, encoding="utf-8").read()
    secciones = re.split(r"^## Sistema: ", texto, flags=re.M)[1:]
    return {s.split("\n", 1)[0].strip(): s.split("\n", 1)[1].strip() for s in secciones}


def construir_contexto(rec, historial, top_sasrec, atencion, top_coseno, auditoria, filtros=None):
    # Contexto estructurado real: solo datos que salen del modelo y de la base (sin identificadores)
    peso = dict(zip(atencion["posición"], atencion["peso"])) if atencion is not None else {}
    ctx = {
        "historial_en_orden": [
            {"orden": k, "juego": rec.nombre(it), "genero": rec.mapa.at[it, "genero_principal"],
             "generos": rec.mapa.at[it, "generos"], "precio_usd": rec.mapa.at[it, "precio"],
             "peso_atencion": round(float(peso.get(k, 0.0)), 3)}
            for k, it in enumerate(historial, 1)],
        "top_sasrec": [
            {"posicion": int(f["#"]), "juego": f["juego"], "genero": f["género"], "precio_usd": f["precio"],
             "probabilidad_pct": round(f["prob. %"], 2), "reseñas": int(f["reseñas"])}
            for _, f in top_sasrec.iterrows()] if top_sasrec is not None else [],
        "top_coseno": [
            {"posicion": int(f["#"]), "juego": f["juego"], "genero": f["género"], "reseñas": int(f["reseñas"])}
            for _, f in top_coseno.iterrows()],
        "auditor": auditoria,
        "notas": {
            "juegos_en_historial": len(historial),
            "arranque_en_frio": len(historial) <= 2,
            "lista_principal": "SASRec" if top_sasrec is not None else "popularidad (sin historial, SASRec no aplica)",
            "filtros_aplicados_al_modelo": filtros or {},
            "contexto_de_la_base": "Reseñas de Steam; el 20 % de los juegos concentra el 78,1 % de las reseñas.",
        },
    }
    return json.loads(json.dumps(ctx, default=lambda x: None if x != x else x))   # NaN -> null


class Analista:
    def __init__(self, api_key, prompts, modelos=None):
        from google import genai
        from google.genai import types
        # Sin reintentos automáticos del SDK: con la cuota gratuita, un error 429 hacía esperar
        # hasta varios minutos. Si un modelo no responde en 45 s, se pasa al siguiente (otra cuota).
        self.cliente = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=45_000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.prompts = prompts
        self.modelos = modelos or MODELOS_POR_DEFECTO

    def _llamar(self, sistema, mensajes, json_salida=False, temperatura=0.2):
        from google.genai import types
        contenidos = [types.Content(role=r, parts=[types.Part.from_text(text=t)]) for r, t in mensajes]
        base = dict(system_instruction=sistema, temperature=temperatura,
                    response_mime_type="application/json" if json_salida else None)
        ultimo_error = None
        inicio = time.time()            # la latencia incluye los intentos fallidos: es lo que espera el usuario
        for modelo in self.modelos:
            # Razonamiento "bajo" para responder rápido; si el modelo no lo admite, se llama sin él
            for extra in ({"thinking_config": types.ThinkingConfig(thinking_level="low")}, {}):
                try:
                    r = self.cliente.models.generate_content(
                        model=modelo, contents=contenidos, config=types.GenerateContentConfig(**base, **extra))
                except Exception as e:
                    ultimo_error = e
                    if extra and "think" in str(e).lower():
                        continue                     # reintenta el mismo modelo sin razonamiento
                    break                            # cuota agotada o modelo inexistente: siguiente modelo
                uso = r.usage_metadata
                return {"texto": r.text or "", "modelo": modelo, "latencia_s": round(time.time() - inicio, 2),
                        "tokens_entrada": getattr(uso, "prompt_token_count", 0) or 0,
                        "tokens_salida": (getattr(uso, "candidates_token_count", 0) or 0)
                                         + (getattr(uso, "thoughts_token_count", 0) or 0)}
        raise RuntimeError(f"Ningún modelo respondió: {ultimo_error}")

    @staticmethod
    def _bloque(contexto):
        return "CONTEXTO:\n```json\n" + json.dumps(contexto, ensure_ascii=False) + "\n```"

    def analisis_inicial(self, contexto):
        pedido = self._bloque(contexto) + "\n\nEscribe el análisis inicial de esta recomendación."
        return self._llamar(self.prompts["analista"], [("user", pedido)])

    def clasificar(self, pregunta, generos_validos):
        sistema = self.prompts["clasificador"].replace("{GENEROS}", ", ".join(generos_validos))
        r = self._llamar(sistema, [("user", pregunta)], json_salida=True, temperatura=0.0)
        try:
            r["decision"] = json.loads(r["texto"])
        except json.JSONDecodeError:
            r["decision"] = {"accion": "responder", "filtros": {}, "motivo": "respuesta no legible"}
        return r

    def responder(self, pregunta, contexto, chat_previo=(), nota=""):
        mensajes = [("user", self._bloque(contexto) + "\n\nEste es el contexto de la conversación.")]
        mensajes.append(("model", "Entendido. Solo hablaré de los juegos de este contexto."))
        for p, rsp in list(chat_previo)[-3:]:           # memoria corta para cuidar la cuota
            mensajes += [("user", p), ("model", rsp)]
        mensajes.append(("user", (nota + "\n\n" if nota else "") + "PREGUNTA: " + pregunta))
        return self._llamar(self.prompts["analista"], mensajes)


# ---------------------------------------------------------------------------
# Verificación automática de honestidad
# ---------------------------------------------------------------------------
def _normalizar(t):
    return re.sub(r"[™®©]", "", str(t)).lower().strip()


def _alfanum(t):
    return re.sub(r"[^a-z0-9]", "", _normalizar(t))


def verificar_honestidad(texto, contexto, catalogo):
    # 1) ¿Nombra algún juego del catálogo que NO estaba en el contexto?
    # 2) ¿Cada "#k Nombre" corresponde de verdad a la posición k de la lista del modelo?
    t = _normalizar(texto)
    en_contexto = {_normalizar(x["juego"]) for clave in ("historial_en_orden", "top_sasrec", "top_coseno")
                   for x in contexto[clave]}
    fuera = []
    for nombre in catalogo:
        n = _normalizar(nombre)
        if len(n) < 6 or n in en_contexto or any(n in c for c in en_contexto):
            continue
        if n in t and re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", t):
            fuera.append(nombre)

    errores = []
    for m in re.finditer(r"(coseno\s*)?#\s?(\d{1,2})\b", t):
        lista = contexto["top_coseno"] if m.group(1) else contexto["top_sasrec"]
        k = int(m.group(2))
        ventana = t[m.end(): m.end() + 80]
        if not lista or k > len(lista):
            errores.append(f"#{k} no existe en la lista")
            continue
        # El prompt exige "#k Nombre": se compara el nombre COMPLETO solo con letras y números,
        # así "Call of Duty WWII" calza con "Call of Duty®: WWII", pero "Counter-Steam" no calza con "Counter-Strike"
        esperado = _alfanum(lista[k - 1]["juego"])
        largo = len(lista[k - 1]["juego"]) + 25
        despues = _alfanum(t[m.end(): m.end() + largo])
        antes = _alfanum(t[max(0, m.start() - largo): m.start()])      # también vale "Nombre (coseno #k)"
        if not despues.startswith(esperado) and esperado not in antes:
            errores.append(f"#{k} no corresponde exactamente a {lista[k - 1]['juego']}")
    return {"ok": not fuera and not errores, "fuera_de_lista": fuera, "rangos_incorrectos": errores}


def buscar_juego(nombre, catalogo):
    # Une lo que escribió el usuario con un nombre real del catálogo (o None)
    minus = {_normalizar(c): c for c in catalogo}
    encontrado = difflib.get_close_matches(_normalizar(nombre), list(minus), n=1, cutoff=0.75)
    return minus[encontrado[0]] if encontrado else None
