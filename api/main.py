"""
Capa 4: Servidor Backend FastAPI Principal con Carga Única de Artefactos (Lifespan).
Proyecto Integrador: TradingJournal NLP (Deriv Volatility 100 / MT5).
Autor: Oscar Cruz (DEV 2).

Cumple con las Secciones 5.4 y 5.5 del PDF del Proyecto Integrador:
- El backend carga una sola vez en memoria los artefactos serializados (.joblib y .json).
- Habilita CORS completo para integración con el Frontend.
- NUNCA reimplementa el pipeline de NLP; delega en el artefacto serializado.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Asegurar que el directorio raíz del proyecto esté en sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore[assignment]

from api.routes import router as api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradingJournalAPI")

# Rutas de artefactos obligatorios del proyecto
MODELS_DIR = ROOT_DIR / "models_prod"
PROD_PIPELINE_PATH = MODELS_DIR / "pipeline_prod.joblib"
METRICS_REPORT_PATH = MODELS_DIR / "metrics_report.json"
HISTORY_PATH = ROOT_DIR / "data" / "processed" / "journal_history.json"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manejador de ciclo de vida de FastAPI (Lifespan):
    Carga en memoria UNA SOLA VEZ al iniciar el servidor todos los modelos,
    reportes y registros de bitácora histórica, garantizando inferencia instantánea.
    """
    logger.info("=== Iniciando Servidor TradingJournal NLP ===")
    logger.info("Directorio raíz: %s", ROOT_DIR)

    # 1. Cargar Pipeline serializado de Producción (LogisticRegression + TF-IDF + Preprocesador)
    if not PROD_PIPELINE_PATH.exists():
        logger.warning(
            "Artefacto de producción no encontrado en %s. "
            "Ejecuta 'python src/models/train.py' para generarlo.",
            PROD_PIPELINE_PATH,
        )
        app.state.pipeline = None
    else:
        logger.info("Cargando pipeline serializado desde %s...", PROD_PIPELINE_PATH)
        if joblib is not None:
            app.state.pipeline = joblib.load(PROD_PIPELINE_PATH)
        else:
            import pickle
            with open(PROD_PIPELINE_PATH, "rb") as f:
                app.state.pipeline = pickle.load(f)
        logger.info("Pipeline cargado exitosamente. Clases: %s", getattr(app.state.pipeline, "classes_", []))

    # 2. Cargar Reporte de Métricas JSON
    app.state.metrics_path = str(METRICS_REPORT_PATH)
    if METRICS_REPORT_PATH.exists():
        try:
            app.state.metrics_report = json.loads(METRICS_REPORT_PATH.read_text(encoding="utf-8"))
            logger.info("Reporte de métricas cargado exitosamente.")
        except Exception as exc:
            logger.error("Error al leer metrics_report.json: %s", exc)
            app.state.metrics_report = {}
    else:
        app.state.metrics_report = {}

    # 3. Cargar Historial de Operaciones para Estadísticas
    app.state.history_path = str(HISTORY_PATH)
    if HISTORY_PATH.exists():
        try:
            app.state.history_records = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            logger.info("Historial cargado: %d registros.", len(app.state.history_records))
        except Exception as exc:
            logger.error("Error al leer journal_history.json: %s", exc)
            app.state.history_records = []
    else:
        app.state.history_records = []

    yield

    logger.info("=== Apagando Servidor TradingJournal NLP ===")


app = FastAPI(
    title="TradingJournal NLP API — Deriv Volatility 100",
    description=(
        "API REST de análisis de bitácoras de trading en español, extracción estructurada "
        "de errores operativos, indicadores técnicos (RSI, MACD, EMA) y clasificación ML."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configuración de CORS para permitir peticiones desde cualquier Frontend (Localhost:3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas bajo el prefijo unificado /api/v1
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["General"])
async def root() -> JSONResponse:
    """Ruta raíz de bienvenida e información del servicio."""
    return JSONResponse(
        content={
            "service": "TradingJournal NLP API",
            "version": "1.0.0",
            "description": "Sistema de Análisis de Bitácoras de Trading y Detección de Errores Operativos en Volatility 100",
            "endpoints": {
                "health": "/api/v1/health",
                "predict": "/api/v1/predict",
                "stats": "/api/v1/stats",
                "metrics": "/api/v1/metrics",
                "docs": "/docs",
            },
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
