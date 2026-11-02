"""
Capa 4 & 5: Pruebas Automatizadas para la API REST y el Pipeline de NLP.
Proyecto Integrador: TradingJournal NLP (Deriv Volatility 100 / MT5).
Autor: Oscar Cruz (DEV 2).

Cumple con las Secciones 5.4, 5.5 y 6.1 del PDF del Proyecto Integrador:
- `test_health_endpoint`: Verifica respuesta HTTP 200 y carga de modelo en memoria.
- `test_predict_oscar_rsi_example`: Prueba el caso insignia acordado con el docente:
  'Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo'.
  Verifica:
  1. Detección de indicador RSI (nivel 28 - sobreventa).
  2. Acción 'Compra'.
  3. Conservación de la negación 'no' en las stop words.
  4. Detección del error operativo 'No esperar cierre de vela'.
  5. Predicción correcta 'Perdedora_Error_Operativo' con alta probabilidad.
- `test_predict_disciplined_win`: Verifica operación disciplinada clasificada como 'Ganadora_Disciplinada'.
- `test_predict_break_even_neutral`: Verifica caso Break-Even clasificado como 'Riesgo_Neutral'.
- `test_validation_empty_and_short_text`: Verifica rechazo HTTP 422 para textos vacíos o de < 3 palabras.
- `test_validation_max_characters`: Verifica rechazo para textos de más de 1000 caracteres.
- `test_stats_endpoint`: Verifica cálculo de errores frecuentes y radar de Volatility 100.
- `test_metrics_endpoint`: Verifica reporte con comparativa LogisticRegression vs Naive Bayes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture que proporciona un TestClient de FastAPI con el ciclo de vida (lifespan) ejecutado."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Verifica que /api/v1/health responda HTTP 200 con el modelo y clases reportadas."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200, f"Error en health check: {response.text}"
    data = response.json()
    assert data["status"] == "ok"
    assert data["production_model"] == "LogisticRegression"
    assert "Ganadora_Disciplinada" in data["classes"]
    assert "Perdedora_Error_Operativo" in data["classes"]
    assert "Riesgo_Neutral" in data["classes"]
    assert data["total_history_records"] >= 0


def test_predict_oscar_rsi_example(client: TestClient) -> None:
    """
    Prueba el caso insignia acordado con el docente:
    'Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo'
    """
    sample_text = (
        "Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo"
    )
    response = client.post("/api/v1/predict", json={"text": sample_text})
    assert response.status_code == 200, f"Error en /predict: {response.text}"
    data = response.json()

    # 1. Verificación de clasificación ML
    assert data["prediction"] == "Perdedora_Error_Operativo", (
        f"Se esperaba 'Perdedora_Error_Operativo' pero se obtuvo '{data['prediction']}'"
    )
    assert data["probabilities"]["Perdedora_Error_Operativo"] > 0.50

    # 2. Verificación de extracción estructurada
    info = data["extracted_trading_info"]
    assert "RSI" in info["indicator"], f"Indicador detectado incorrecto: {info['indicator']}"
    assert "28" in info["indicator"], f"Nivel numérico de RSI no detectado: {info['indicator']}"
    assert "Compra" in info["action"]
    assert "No esperar cierre" in info["operational_error"]

    # 3. Verificación de pipeline NLP y conservación de negaciones
    pipeline_steps = data["pipeline_steps"]
    assert "no" in pipeline_steps["kept_negations"], (
        "La negación 'no' debe conservarse obligatoriamente para detectar el error operativo"
    )
    assert "no" in pipeline_steps["stopwords_removed"], (
        "El token 'no' debe permanecer en la lista filtrada de stop words"
    )

    # 4. Verificación de top términos TF-IDF
    top_terms = [item["term"] for item in pipeline_steps["top_tfidf_features"]]
    assert any("no" in term or "esperar" in term or "rsi" in term for term in top_terms), (
        f"Bigramas clave de trading no encontrados en TF-IDF: {top_terms}"
    )


def test_predict_disciplined_win(client: TestClient) -> None:
    """Verifica que una bitácora disciplinada con confirmación y TP sea clasificada como Ganadora."""
    sample_text = (
        "Operé Volatility 100 en compra tras cruce de EMA 20 y confirmación de vela alcista con RSI en 32, "
        "respeté el take profit y salí en ganancia"
    )
    response = client.post("/api/v1/predict", json={"text": sample_text})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Ganadora_Disciplinada"
    assert data["probabilities"]["Ganadora_Disciplinada"] > 0.50
    assert "Compra" in data["extracted_trading_info"]["action"]


def test_predict_break_even_neutral(client: TestClient) -> None:
    """Verifica que una operación cerrada en Break-Even sea clasificada como Riesgo_Neutral."""
    sample_text = (
        "Entré en compra en Volatility 75 sobre soporte con bandas de Bollinger estrechas "
        "pero el mercado quedó lateral y cerré en break even sin pérdidas"
    )
    response = client.post("/api/v1/predict", json={"text": sample_text})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Riesgo_Neutral"
    assert data["probabilities"]["Riesgo_Neutral"] > 0.40


def test_validation_empty_and_short_text(client: TestClient) -> None:
    """Verifica que la API rechace textos vacíos o de menos de 3 palabras con HTTP 422."""
    # Texto vacío
    r_empty = client.post("/api/v1/predict", json={"text": ""})
    assert r_empty.status_code == 422, "Texto vacío debe retornar HTTP 422"

    # Solo espacios
    r_spaces = client.post("/api/v1/predict", json={"text": "    "})
    assert r_spaces.status_code == 422, "Texto con solo espacios debe retornar HTTP 422"

    # Menos de 3 palabras (1 palabra)
    r_one_word = client.post("/api/v1/predict", json={"text": "Compré"})
    assert r_one_word.status_code == 422, "Texto de 1 palabra debe retornar HTTP 422"

    # Menos de 3 palabras (2 palabras)
    r_two_words = client.post("/api/v1/predict", json={"text": "Compré RSI"})
    assert r_two_words.status_code == 422, "Texto de 2 palabras debe retornar HTTP 422"


def test_validation_max_characters(client: TestClient) -> None:
    """Verifica que la API rechace textos que excedan los 1000 caracteres con HTTP 422."""
    long_text = "compra rsi vela " * 150  # Más de 1500 caracteres
    response = client.post("/api/v1/predict", json={"text": long_text})
    assert response.status_code == 422, "Texto de más de 1000 caracteres debe retornar HTTP 422"


def test_stats_and_metrics_endpoints(client: TestClient) -> None:
    """Verifica que los endpoints /api/v1/stats y /api/v1/metrics devuelvan sus estructuras completas."""
    # 1. Endpoint /stats
    r_stats = client.get("/api/v1/stats")
    assert r_stats.status_code == 200
    stats = r_stats.json()
    assert stats["total_trades"] > 0
    assert "Ganadora_Disciplinada" in stats["outcome_distribution"]
    assert "Perdedora_Error_Operativo" in stats["outcome_distribution"]
    assert len(stats["operational_errors"]) > 0
    assert "volatility_100_summary" in stats
    assert "top_error" in stats["volatility_100_summary"]

    # 2. Endpoint /metrics
    r_metrics = client.get("/api/v1/metrics")
    assert r_metrics.status_code == 200
    metrics = r_metrics.json()
    assert "models" in metrics
    assert "LogisticRegression" in metrics["models"]
    assert "MultinomialNB" in metrics["models"]
    assert "ablation_experiments" in metrics
    assert "error_analysis" in metrics
    assert len(metrics["error_analysis"]) == 3
