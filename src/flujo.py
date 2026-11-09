# Flujo compartido por la app y el notebook 2: así las pruebas de honestidad evalúan
# exactamente lo mismo que ve el usuario en el dashboard.
from src.analista import buscar_juego, construir_contexto, verificar_honestidad
from src.auditor import auditar


def calcular(rec, historial, n=10, filtros=None):
    # El Transformer genera la lista; con historial vacío no aplica y se usa popularidad (declarado)
    if historial:
        top_s, atencion = rec.sasrec(historial, n, filtros)
        top_c = rec.coseno(historial, n, filtros)
    else:
        top_s, atencion = None, None
        top_c = rec.popularidad(historial, n, filtros)
    top_p = rec.popularidad(historial, n, filtros)
    principal = top_s if top_s is not None else top_p
    auditoria = {"sasrec" if top_s is not None else "popularidad": auditar(principal["item"], rec),
                 "coseno": auditar(top_c["item"], rec), "popularidad": auditar(top_p["item"], rec)}
    contexto = construir_contexto(rec, historial, top_s, atencion, top_c, auditoria, filtros)
    return {"top_s": top_s, "atencion": atencion, "top_c": top_c, "top_p": top_p,
            "auditoria": auditoria, "contexto": contexto}


def preguntar(analista, rec, historial, pregunta, n, generos_validos, chat_previo=()):
    # 1) el clasificador decide si basta el contexto o hay que reconsultar al modelo;
    # 2) si hay que reconsultar, SASRec genera la nueva lista (con filtros o con el historial extendido);
    # 3) el analista responde sobre ESE contexto y se verifica la respuesta.
    llamadas = []
    c = analista.clasificar(pregunta, generos_validos)
    llamadas.append(("clasificador", c))
    decision = c["decision"]
    res = calcular(rec, historial, n)
    nota, nuevo = "", None
    if decision.get("accion") == "reconsultar":
        bruto = decision.get("filtros") or {}
        filtros = {k: v for k, v in bruto.items() if k != "agregar_al_historial" and v not in (None, [], "")}
        agregados = [buscar_juego(x, rec.mapa["nombre"]) for x in bruto.get("agregar_al_historial", []) or []]
        agregados = [rec.nombre_a_item[a] for a in agregados if a]
        if filtros or agregados:
            nuevo = calcular(rec, list(historial) + agregados, n, filtros or None)
            res = nuevo
            partes = [f"filtros {filtros}" if filtros else "",
                      f"historial + {[rec.nombre(a) for a in agregados]}" if agregados else ""]
            nota = ("La app volvió a consultar al modelo con " + " y ".join(p for p in partes if p) +
                    ". El CONTEXTO ya es la nueva lista del modelo: analízala.")
    r = analista.responder(pregunta, res["contexto"], chat_previo, nota)
    llamadas.append(("chat", r))
    r["verificacion"] = verificar_honestidad(r["texto"], res["contexto"], rec.mapa["nombre"])
    return {"respuesta": r, "decision": decision, "reconsultado": bool(nota), "resultado": res,
            "llamadas": llamadas}
