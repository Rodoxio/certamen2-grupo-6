# Recomendador: carga SASRec (entrenado en el notebook 1) y reconstruye las líneas base
# con los mismos datos de entrenamiento. Nada se entrena aquí: solo inferencia en CPU.
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse


# ---------------------------------------------------------------------------
# Arquitectura (idéntica a la sección 7 del notebook 1)
# ---------------------------------------------------------------------------
class BloqueTransformer(nn.Module):
    # Atención causal + feed-forward, cada una con normalización previa y conexión residual
    def __init__(self, d, cabezas, dropout):
        super().__init__()
        self.norma_atencion = nn.LayerNorm(d)
        self.atencion = nn.MultiheadAttention(d, cabezas, dropout=dropout, batch_first=True)
        self.norma_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mascara, guardar_pesos=False):
        h = self.norma_atencion(x)
        salida, pesos = self.atencion(h, h, h, attn_mask=mascara,
                                      need_weights=guardar_pesos, average_attn_weights=True)
        x = x + self.dropout(salida)
        x = x + self.dropout(self.ffn(self.norma_ffn(x)))
        return x, pesos


class SASRec(nn.Module):
    def __init__(self, n_items, largo=50, d=64, bloques=2, cabezas=1, dropout=0.2):
        super().__init__()
        self.d, self.cabezas = d, cabezas
        self.items = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.posiciones = nn.Embedding(largo, d)
        self.dropout = nn.Dropout(dropout)
        self.bloques = nn.ModuleList([BloqueTransformer(d, cabezas, dropout) for _ in range(bloques)])
        self.norma_final = nn.LayerNorm(d)
        self.pesos_atencion = None

    def codificar(self, seq, guardar_pesos=False):
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0)
        es_item = (seq != 0).unsqueeze(-1)
        x = self.items(seq) * self.d ** 0.5 + self.posiciones(pos)
        x = self.dropout(x) * es_item
        futuro = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)
        relleno = (seq == 0).unsqueeze(1)
        diagonal = torch.eye(L, dtype=torch.bool, device=seq.device)
        mascara = (futuro.unsqueeze(0) | relleno) & ~diagonal
        mascara = mascara.repeat_interleave(self.cabezas, dim=0)
        for bloque in self.bloques:
            x, pesos = bloque(x, mascara, guardar_pesos)
            x = x * es_item
        self.pesos_atencion = pesos
        return self.norma_final(x)

    def puntajes(self, seq, guardar_pesos=False):
        return self.codificar(seq, guardar_pesos)[:, -1, :] @ self.items.weight.T


def rellenar(h, largo):
    # Últimos `largo` ítems con 0 al inicio (igual que en el notebook)
    h = list(h)[-largo:]
    return [0] * (largo - len(h)) + h


# ---------------------------------------------------------------------------
# Recomendador con las tres fuentes: SASRec, ítem–ítem coseno y popularidad
# ---------------------------------------------------------------------------
class Recomendador:
    GAMMA = 0.9   # decaimiento del coseno elegido con validación en el notebook 1

    def __init__(self, carpeta="."):
        torch.set_num_threads(2)
        self.mapa = pd.read_csv(os.path.join(carpeta, "data", "mapa_items.csv")).set_index("item")
        self.mapa["generos"] = self.mapa["generos"].fillna("").str.split("|")
        self.metricas = pd.read_csv(os.path.join(carpeta, "data", "metricas.csv"))
        secuencias = pd.read_parquet(os.path.join(carpeta, "data", "secuencias.parquet"))
        self.secuencias = [list(map(int, s)) for s in secuencias["secuencia"]]

        guardado = torch.load(os.path.join(carpeta, "modelos", "sasrec.pt"), map_location="cpu")
        self.N, self.largo, self.config = guardado["n_items"], guardado["largo"], guardado["config"]
        self.modelo = SASRec(self.N, self.largo, **self.config)
        self.modelo.load_state_dict(guardado["estado"])
        self.modelo.eval()

        # Líneas base calculadas SOLO con la parte de entrenamiento (todo menos los 2 últimos juegos)
        entrena = [s[:-2] for s in self.secuencias]
        self.conteo = np.zeros(self.N + 1)
        np.add.at(self.conteo, np.concatenate(entrena), 1)
        self.n_usuarios = len(entrena)

        filas = np.repeat(np.arange(len(entrena)), [len(s) for s in entrena])
        X = sparse.csr_matrix((np.ones(len(filas), dtype=np.float32), (filas, np.concatenate(entrena))),
                              shape=(len(entrena), self.N + 1))
        # Coseno por bloques de columnas, escribiendo directo en float16 (~90 MB en total):
        # así la app no supera la memoria de Streamlit Cloud.
        X = X.tocsc()
        norma = np.sqrt(np.asarray(X.sum(axis=0)).ravel()) + 1e-9
        self.similitud = np.zeros((self.N + 1, self.N + 1), dtype=np.float16)
        XT = X.T.tocsr()
        for ini in range(0, self.N + 1, 512):
            fin = min(ini + 512, self.N + 1)
            bloque = (XT @ X[:, ini:fin]).toarray().astype(np.float32)
            bloque /= norma[:, None]
            bloque /= norma[None, ini:fin]
            self.similitud[:, ini:fin] = bloque
        np.fill_diagonal(self.similitud, 0)

        self.desempate = self.conteo / (self.conteo.max() + 1) * 1e-3
        self.cabeza = set(np.argsort(-self.conteo)[: int(0.2 * self.N)].tolist())   # 20 % más popular
        self.nombre_a_item = {n: i for i, n in self.mapa["nombre"].items()}

    # ---------------- utilidades ----------------
    def nombre(self, it):
        return self.mapa.at[it, "nombre"]

    def _tabla(self, items, puntajes, columna_puntaje):
        filas = []
        for pos, (it, p) in enumerate(zip(items, puntajes), 1):
            fila = self.mapa.loc[it]
            filas.append({"#": pos, "item": int(it), "juego": fila["nombre"], "género": fila["genero_principal"],
                          "precio": fila["precio"], "reseñas": int(self.conteo[it]), columna_puntaje: float(p)})
        return pd.DataFrame(filas)

    def _mascara(self, historial, filtros=None):
        # True = el juego NO puede recomendarse (relleno, ya jugados o fuera de los filtros)
        bloqueado = np.zeros(self.N + 1, dtype=bool)
        bloqueado[0] = True
        bloqueado[list(historial)] = True
        if filtros:
            g = self.mapa["generos"].reindex(range(self.N + 1))
            if filtros.get("excluir_generos"):
                excl = set(filtros["excluir_generos"])
                bloqueado |= np.array([isinstance(x, list) and bool(excl & set(x)) for x in g])
            if filtros.get("solo_generos"):
                solo = set(filtros["solo_generos"])
                bloqueado |= ~np.array([isinstance(x, list) and bool(solo & set(x)) for x in g])
            if filtros.get("precio_max") is not None:
                precio = self.mapa["precio"].reindex(range(self.N + 1)).values
                bloqueado |= ~(np.nan_to_num(precio, nan=np.inf) <= float(filtros["precio_max"]))
        return bloqueado

    # ---------------- modelos ----------------
    def sasrec(self, historial, n=10, filtros=None):
        # Devuelve el Top-N de SASRec (orden del modelo) y los pesos de atención sobre el historial.
        # Los filtros solo ocultan juegos: el orden de los que quedan es el del modelo.
        h = list(historial)[-self.largo:]
        entrada = torch.tensor([rellenar(h, self.largo)])
        with torch.no_grad():
            p = self.modelo.puntajes(entrada, guardar_pesos=True)[0].numpy().astype(np.float64)
        pesos = self.modelo.pesos_atencion[0, -1].numpy()[-len(h):]
        bloqueado = self._mascara(historial, filtros)
        p[bloqueado] = -np.inf
        prob = np.exp(p - p[~bloqueado].max())
        prob = prob / prob[~bloqueado].sum()
        top = np.argsort(-p)[:n]
        top = top[np.isfinite(p[top])]
        tabla = self._tabla(top, prob[top] * 100, "prob. %")
        atencion = pd.DataFrame({"posición": range(1, len(h) + 1), "item": h,
                                 "juego": [self.nombre(i) for i in h], "peso": pesos})
        return tabla, atencion

    def coseno(self, historial, n=10, filtros=None):
        h = list(historial)[-self.largo:]
        pesos = np.array([self.GAMMA ** (len(h) - 1 - k) for k in range(len(h))])
        p = (pesos[:, None] * self.similitud[h].astype(np.float32)).sum(0).astype(np.float64) + self.desempate
        p[self._mascara(historial, filtros)] = -np.inf
        top = np.argsort(-p)[:n]
        return self._tabla(top, p[top], "puntaje")

    def popularidad(self, historial, n=10, filtros=None):
        p = self.conteo.astype(np.float64).copy()
        p[self._mascara(historial, filtros)] = -np.inf
        top = np.argsort(-p)[:n]
        return self._tabla(top, p[top], "reseñas en entrenamiento")

    # ---------------- ejemplos de la base (los 3 usuarios de la sección 8.5) ----------------
    def ejemplos(self):
        rng = np.random.default_rng(2026)
        largos = np.array([len(s) - 1 for s in self.secuencias])
        grupos = [("corto", (largos >= 4) & (largos <= 6)), ("medio", (largos >= 10) & (largos <= 15)),
                  ("largo", (largos >= 25) & (largos <= self.largo))]
        salida = {}
        for etiqueta, cond in grupos:
            u = int(rng.choice(np.where(cond)[0]))
            salida[etiqueta] = {"historial": self.secuencias[u][:-1], "siguiente_real": self.secuencias[u][-1]}
        return salida
