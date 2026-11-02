"""
Capa 4: Esquemas y Validaciones Pydantic para la API REST FastAPI.
Proyecto Integrador: TradingJournal NLP (Deriv Volatility 100 / MT5).
Autor: Oscar Cruz (DEV 2).

Cumple con las Secciones 5.5 del PDF del Proyecto Integrador:
- Validación robusta de entradas: rechaza texto vacío, menos de 3 palabras o más de 1000 caracteres.
- Esquemas estructurados para predicción, desglose del pipeline NLP Antes/Después,
  entidades de trading (RSI, error operativo, acción), estadísticas acumuladas y métricas ML.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# CONFIGURACIÓN DE LÍMITES DE VALIDACIÓN (MODIFICACIÓN EN VIVO DEFENSA)
# ==============================================================================
# Si el Ing. Kenji pide en vivo: "Cambia la validación para rechazar menos de 5 palabras"
# Solo se modifica MIN_WORDS = 5 aquí en 10 segundos.
MIN_WORDS: int = 3
MAX_CHARS: int = 1000


# ==============================================================================
# ESQUEMAS DE ENTRADA (REQUESTS)
# ==============================================================================

class PredictionRequest(BaseModel):
    """Esquema de entrada para el análisis de una bitácora de trading."""
    text: str = Field(
        ...,
        description="Texto en español de la bitácora o estrategia de trading a analizar.",
        examples=[
            "Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo"
        ],
    )

    @field_validator("text")
    @classmethod
    def validate_trading_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("El registro de trading debe ser una cadena de texto.")

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("El registro de trading no puede estar vacío ni contener solo espacios.")

        if len(cleaned) > MAX_CHARS:
            raise ValueError(f"El texto no debe exceder los {MAX_CHARS} caracteres (longitud actual: {len(cleaned)}).")

        words = cleaned.split()
        if len(words) < MIN_WORDS:
            raise ValueError(
                f"El registro de trading debe contener al menos {MIN_WORDS} palabras para un análisis NLP significativo (palabras ingresadas: {len(words)})."
            )

        return cleaned


# ==============================================================================
# ESQUEMAS DE EXTRACCIÓN Y PIPELINE NLP (RESPONSES)
# ==============================================================================

class ExtractedTradingInfo(BaseModel):
    """Información estructurada extraída automáticamente de la bitácora."""
    indicator: str = Field(..., description="Indicador técnico y nivel (ej. 'RSI (Nivel 28 - Sobreventa)')")
    indicator_code: str = Field(..., description="Código normalizado del indicador (ej. 'RSI', 'EMA', 'MACD')")
    action: str = Field(..., description="Acción operativa ejecutada (ej. 'Compra (BUY)' o 'Venta (SELL)')")
    market: str = Field(..., description="Activo sintético o mercado (ej. 'Volatility 100', 'Crash 500')")
    timeframe: str = Field(..., description="Temporalidad detectada (ej. 'M1', 'M5', 'H1' o 'No especificada')")
    operational_error: str = Field(..., description="Error operativo detectado (ej. 'No esperar cierre de vela')")
    outcome_detected: str = Field(..., description="Resultado o desenlace mencionado en el texto")


class TopTfidfFeature(BaseModel):
    """Término o bigrama con peso TF-IDF activo en el registro."""
    term: str = Field(..., description="Término o bigrama unificado")
    weight: float = Field(..., description="Peso TF-IDF ponderado")


class NLPStepBreakdown(BaseModel):
    """Demostración visual Antes / Después de cada fase del pipeline de NLP."""
    raw_text: str = Field(..., description="Texto original tal como fue ingresado")
    normalized: str = Field(..., description="Paso 1: Normalizado (minúsculas, URLs eliminadas, tokens semánticos de RSI y emojis)")
    tokens: list[str] = Field(..., description="Paso 2: Tokenización en palabras del dominio de trading")
    stopwords_removed: list[str] = Field(..., description="Paso 3: Tokens útiles tras filtrar stop words")
    removed_stopwords_list: list[str] = Field(..., description="Stop words descartadas")
    kept_negations: list[str] = Field(..., description="Palabras de negación u omisión conservadas obligatoriamente ('no', 'sin', etc.)")
    lemmatized_or_stemmed: list[str] = Field(..., description="Paso 4: Lemas o raíces canónicas resultantes")
    top_tfidf_features: list[TopTfidfFeature] = Field(..., description="Paso 5: Top términos y bigramas con mayor peso TF-IDF")
    active_mode: str = Field(..., description="Modo morfológico activo ('lemma' o 'stem')")


class PredictionResponse(BaseModel):
    """Respuesta completa del endpoint `/api/v1/predict`."""
    status: str = Field(default="success")
    text: str = Field(..., description="Texto analizado")
    prediction: str = Field(..., description="Clase diagnóstica asignada por el modelo ML")
    probabilities: dict[str, float] = Field(..., description="Distribución de probabilidades por clase")
    confidence: float = Field(..., description="Probabilidad de la clase predicha (0.0 a 1.0)")
    extracted_trading_info: ExtractedTradingInfo
    pipeline_steps: NLPStepBreakdown
    inference_time_ms: float = Field(..., description="Tiempo de inferencia y extracción en milisegundos")
    saved_to_history: bool = Field(default=True, description="Indica si se persistió en el historial estadístico")


# ==============================================================================
# ESQUEMAS DE ESTADÍSTICAS, RADAR Y ESTADO DEL SISTEMA
# ==============================================================================

class OperationalErrorStat(BaseModel):
    error: str
    count: int
    percentage: float


class IndicatorStat(BaseModel):
    total_trades: int
    wins: int
    losses: int
    neutral: int
    winrate: float


class Volatility100Radar(BaseModel):
    total_trades: int
    winrate: float
    top_error: str
    recommendation: str


class TradingStatsResponse(BaseModel):
    status: str = Field(default="success")
    total_trades: int
    global_winrate: float
    outcome_distribution: dict[str, int]
    top_fatal_error: str
    operational_errors: list[OperationalErrorStat]
    indicator_performance: dict[str, IndicatorStat]
    market_performance: dict[str, Any]
    volatility_100_summary: Volatility100Radar


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    project: str = Field(default="TradingJournal NLP — Deriv Volatility 100")
    production_model: str = Field(default="LogisticRegression")
    baseline_model: str = Field(default="MultinomialNB")
    nlp_mode: str = Field(default="lemma")
    classes: list[str]
    total_history_records: int
