"""
Capa 4: Rutas y Controladores de la API REST FastAPI.
Proyecto Integrador: TradingJournal NLP (Deriv Volatility 100 / MT5).
Autor: Oscar Cruz (DEV 2).

Cumple con las Secciones 5.4 y 5.5 del PDF del Proyecto Integrador:
- `GET /health`: Estado del servicio, versión y modelo cargado.
- `POST /predict`: Inferencia en tiempo real, desglose paso a paso de NLP,
  extracción estructurada (RSI, error, acción) y persistencia en historial.
- `GET /stats`: Estadísticas dinámicas acumuladas de errores y winrate por indicador.
- `GET /metrics`: Comparativa de modelos (LogisticRegression vs Naive Bayes),
  ablación de NLP (Stemming vs Lemma, N-gramas), matrices de confusión y análisis de errores.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas import (
    ExtractedTradingInfo,
    HealthResponse,
    NLPStepBreakdown,
    PredictionRequest,
    PredictionResponse,
    TopTfidfFeature,
    TradingStatsResponse,
)
from src.models.evaluate import compute_trading_statistics
from src.nlp.vectorizer import extract_top_tfidf_features

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Monitoreo"])
async def get_health(request: Request) -> HealthResponse:
    """Verifica el estado del servicio y reporta el modelo en producción cargado en memoria."""
    pipeline = getattr(request.app.state, "pipeline", None)
    history = getattr(request.app.state, "history_records", [])
    classes = [str(c) for c in getattr(pipeline, "classes_", [
        "Ganadora_Disciplinada",
        "Riesgo_Neutral",
        "Perdedora_Error_Operativo",
    ])]
    nlp_mode = getattr(pipeline.named_steps.get("preprocessor"), "mode", "lemma") if pipeline else "lemma"

    return HealthResponse(
        status="ok",
        version="1.0.0",
        project="TradingJournal NLP — Deriv Volatility 100",
        production_model="LogisticRegression",
        baseline_model="MultinomialNB",
        nlp_mode=nlp_mode,
        classes=classes,
        total_history_records=len(history),
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Inferencia NLP & ML"])
async def predict_trading_journal(request: Request, payload: PredictionRequest) -> PredictionResponse:
    """
    Procesa un registro de bitácora de trading a través de las 5 capas:
    1. Valida entrada (rechaza vacío, < 3 palabras, > 1000 caracteres).
    2. Ejecuta preprocesamiento y genera explicación Antes/Después.
    3. Extrae entidades estructuradas (indicador RSI, acción, mercado, error operativo).
    4. Clasifica con el modelo de ML (LogisticRegression) y extrae probabilidades.
    5. Guarda el registro estructurado para actualizar las estadísticas dinámicas.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de Machine Learning no se encuentra cargado en el servidor.",
        )

    t0 = time.perf_counter()

    try:
        # Regla de oro del PDF: Usar el preprocesador del pipeline serializado, nunca reimplementarlo
        preprocessor = pipeline.named_steps["preprocessor"]
        steps = preprocessor.explain_steps(payload.text)
        entities = preprocessor.extract_trading_entities(payload.text)

        # Inferencia con el modelo ML
        pred_label = str(pipeline.predict([payload.text])[0])

        prob_dict: dict[str, float] = {}
        confidence: float = 1.0
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([payload.text])[0]
            classes = [str(c) for c in getattr(pipeline, "classes_", [])]
            for cls_name, p in zip(classes, probs):
                prob_dict[cls_name] = round(float(p), 4)
            confidence = prob_dict.get(pred_label, 1.0)
        else:
            prob_dict = {pred_label: 1.0}

        # Extracción de top características TF-IDF activas en la bitácora
        lemmas_text = " ".join(steps["lemmatized_or_stemmed"])
        raw_top_tfidf = extract_top_tfidf_features(pipeline, lemmas_text, top_k=8)
        top_tfidf = [TopTfidfFeature(term=item["term"], weight=item["weight"]) for item in raw_top_tfidf]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Persistir la operación analizada en el historial estadístico
        history_records = getattr(request.app.state, "history_records", None)
        history_path = getattr(request.app.state, "history_path", None)

        saved = False
        if history_records is not None and history_path is not None:
            new_record = {
                "id": len(history_records) + 1,
                "text": payload.text,
                "prediction": pred_label,
                "indicator": entities.get("indicator", "Price_Action"),
                "action": entities.get("action", "General"),
                "market": entities.get("market", "Volatility 100"),
                "operational_error": entities.get("operational_error", "Ninguno"),
                "timestamp": datetime.now().isoformat(),
            }
            history_records.append(new_record)
            saved = True
            try:
                Path(history_path).write_text(
                    json.dumps(history_records, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                # No detener la inferencia si falla la escritura en disco
                pass

        return PredictionResponse(
            status="success",
            text=payload.text,
            prediction=pred_label,
            probabilities=prob_dict,
            confidence=confidence,
            extracted_trading_info=ExtractedTradingInfo(
                indicator=entities.get("indicator", "Price_Action"),
                indicator_code=entities.get("indicator_code", "Price_Action"),
                action=entities.get("action", "General"),
                market=entities.get("market", "Volatility 100"),
                timeframe=entities.get("timeframe", "No especificada"),
                operational_error=entities.get("operational_error", "Ninguno"),
                outcome_detected=entities.get("outcome_detected", "No especificado"),
            ),
            pipeline_steps=NLPStepBreakdown(
                raw_text=steps["raw_text"],
                normalized=steps["normalized"],
                tokens=steps["tokens"],
                stopwords_removed=steps["stopwords_removed"],
                removed_stopwords_list=steps["removed_stopwords_list"],
                kept_negations=steps["kept_negations"],
                lemmatized_or_stemmed=steps["lemmatized_or_stemmed"],
                top_tfidf_features=top_tfidf,
                active_mode=steps.get("active_mode", "lemma"),
            ),
            inference_time_ms=elapsed_ms,
            saved_to_history=saved,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el procesamiento del pipeline: {str(exc)}",
        ) from exc


@router.get("/stats", response_model=TradingStatsResponse, tags=["Estadísticas de Trading"])
async def get_trading_statistics(request: Request) -> TradingStatsResponse:
    """
    Retorna las estadísticas acumuladas de todas las bitácoras analizadas:
    - Frecuencia de cada error operativo (FOMO, No esperar cierre de vela, etc.).
    - Winrate por indicador técnico (RSI, MACD, EMA).
    - Métricas especializadas para Deriv Volatility 100 con recomendación preventiva.
    """
    history = getattr(request.app.state, "history_records", [])
    stats = compute_trading_statistics(history)
    return TradingStatsResponse(**stats)


@router.get("/metrics", tags=["Métricas de Machine Learning"])
async def get_model_metrics(request: Request) -> dict[str, Any]:
    """
    Devuelve el reporte completo de experimentación serializado en `models_prod/metrics_report.json`:
    - Comparativa `LogisticRegression` vs `MultinomialNB` (5-fold CV y Test 20%).
    - Ablación de NLP (Stemming vs Lematización, Unigramas vs Bigramas).
    - Matrices de Confusión por modelo.
    - Análisis de 3 errores reales del clasificador con explicación cualitativa.
    """
    metrics = getattr(request.app.state, "metrics_report", None)
    if metrics is None:
        metrics_path = getattr(request.app.state, "metrics_path", None)
        if metrics_path and Path(metrics_path).exists():
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="El reporte de métricas no se encuentra disponible.",
            )
    return metrics
