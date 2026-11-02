"""
Capa 2: Vectorización TF-IDF con Bigramas y Extracción de Pesos Activos.
Proyecto Integrador: TradingJournal NLP (Deriv Volatility 100 / MT5).
Autor: Oscar Cruz (DEV 2).

Cumple con las Secciones 5.2 y 5.4 del PDF del Proyecto Integrador:
1. `build_tfidf_vectorizer`: Configuración formal y documentada del vectorizador TF-IDF
   con soporte de unigramas y bigramas `(1, 2)`, `sublinear_tf=True`, `min_df=2`,
   `max_df=0.85` y `max_features=3000`.
2. `extract_top_tfidf_features`: Extractor de términos y bigramas con mayor peso
   en tiempo real para una bitácora analizada, para alimentar la vista explicativa
   del Frontend y la API FastAPI.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    # Respaldo si scikit-learn no estuviese disponible en algún entorno mínimo
    TfidfVectorizer = None  # type: ignore[assignment]


def build_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 3000,
    min_df: int | float = 2,
    max_df: float = 0.85,
    sublinear_tf: bool = True,
) -> Any:
    """
    Construye un vectorizador TF-IDF con los hiperparámetros optimizados para bitácoras de trading.

    Justificación técnica (Pregunta 1 del docente y Sección 5.2 del PDF):
    - `ngram_range=(1, 2)`: Los unigramas sueltos ("no", "esperar", "cierre") pierden el contexto
      operativo; el bigrama ("no esperar", "cierre vela", "sin stop") captura la indisciplina exacta.
    - `max_features=3000`: Acota el vocabulario a los 3,000 términos y combinaciones más discriminativas,
      evitando la maldición de dimensionalidad.
    - `min_df=2`: Ignora términos que aparecen en 1 sola bitácora (errores tipográficos aislados).
    - `max_df=0.85`: Descarta palabras omnipresentes que aparecen en más del 85% del corpus.
    - `sublinear_tf=True`: Aplica escala logarítmica (1 + log(TF)), evitando que la repetición
      excesiva de un término ("perder perder perder") distorsione desproporcionadamente el vector.
    """
    if TfidfVectorizer is not None:
        return TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
        )
    
    # Importación dinámica del fallback en caso extremo
    from src.models.train import SparseTfidfVectorizer
    return SparseTfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df if isinstance(min_df, int) else 2,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )


def extract_top_tfidf_features(
    pipeline_or_vectorizer: Any,
    processed_text: str,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Extrae los términos y bigramas con mayor peso TF-IDF para un texto preprocesado.

    Parámetros:
    -----------
    pipeline_or_vectorizer:
        Pipeline entrenado (`sklearn.pipeline.Pipeline` que contenga el paso 'tfidf' o 'vectorizer')
        o directamente una instancia entrenada de `TfidfVectorizer` / `SparseTfidfVectorizer`.
    processed_text:
        Texto ya preprocesado (tokens lematizados separados por espacio, ej.
        "entrar compra rsi 28 no esperar cierre vela perder").
    top_k:
        Cantidad máxima de términos más relevantes a retornar.

    Retorna:
    --------
    Lista de diccionarios ordenados descendentemente por peso:
    [
        {"term": "no esperar", "weight": 0.462},
        {"term": "cierre vela", "weight": 0.418},
        ...
    ]
    """
    if not processed_text or not isinstance(processed_text, str):
        return []

    # Extraer el vectorizador si se pasó un Pipeline de sklearn o Pipeline personalizado
    vectorizer = pipeline_or_vectorizer
    if hasattr(pipeline_or_vectorizer, "named_steps"):
        named_steps = pipeline_or_vectorizer.named_steps
        if "tfidf" in named_steps:
            vectorizer = named_steps["tfidf"]
        elif "vectorizer" in named_steps:
            vectorizer = named_steps["vectorizer"]
        else:
            for step in named_steps.values():
                if hasattr(step, "transform") and hasattr(step, "vocabulary_"):
                    vectorizer = step
                    break
    elif hasattr(pipeline_or_vectorizer, "steps"):
        for _, step in pipeline_or_vectorizer.steps:
            if hasattr(step, "transform") and hasattr(step, "vocabulary_"):
                vectorizer = step
                break

    if not hasattr(vectorizer, "transform") or not hasattr(vectorizer, "vocabulary_"):
        return []

    if not vectorizer.vocabulary_:
        return []

    # Transformar el texto con el vectorizador entrenado
    try:
        vec = vectorizer.transform([processed_text])
    except Exception:
        return []

    # Obtener los nombres de características
    if hasattr(vectorizer, "get_feature_names_out"):
        feature_names = vectorizer.get_feature_names_out()
    elif hasattr(vectorizer, "feature_names_"):
        feature_names = vectorizer.feature_names_
    elif hasattr(vectorizer, "get_feature_names"):
        feature_names = vectorizer.get_feature_names()
    else:
        # Reconstruir nombres desde el vocabulario ordenado por índice
        inv_vocab = {idx: term for term, idx in vectorizer.vocabulary_.items()}
        feature_names = [inv_vocab[i] for i in range(len(inv_vocab))]

    # Obtener índices y valores no nulos
    if hasattr(vec, "tocoo"):
        coo = vec.tocoo()
        col_indices = coo.col
        weights = coo.data
    else:
        # Si es un array denso
        row = np.asarray(vec).ravel()
        col_indices = np.nonzero(row)[0]
        weights = row[col_indices]

    if len(col_indices) == 0:
        return []

    # Construir lista de términos y pesos
    features: list[dict[str, Any]] = []
    for col_idx, weight in zip(col_indices, weights):
        term = str(feature_names[col_idx])
        features.append({
            "term": term,
            "weight": round(float(weight), 4),
        })

    # Ordenar de mayor a menor peso
    features.sort(key=lambda x: (-x["weight"], x["term"]))
    return features[:top_k]
