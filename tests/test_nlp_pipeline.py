"""
Suite de Pruebas Unitarias y de QA para el Pipeline NLP y Modelos de Trading.
Proyecto Integrador: TradingJournal NLP (Deriv Volatility 100 / MT5).
Autores: DEV 1 (Maykol) & DEV 2 (Oscar Cruz).

Cobertura de QA:
1. Normalización de dominios y niveles técnicos de RSI (<30 y >70).
2. Traducción de emojis operativos a tokens semánticos (_emoji_ganancia_, _emoji_perdida_).
3. Conservación estricta de palabras de negación ("no", "sin", "antes", etc.) en Stop Words.
4. Lematización morfológica en español (conjugaciones a infinitivo base).
5. Extractor estructurado de entidades de trading (indicadores, acciones, errores, activos sintéticos).
6. Construcción y extracción de bigramas TF-IDF con pesos activos.
7. Cálculo exacto de métricas de clasificación, matrices de confusión y error_rate.
8. Generación de estadísticas de trading y Radar preventivo para Deriv Volatility 100.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
import pytest

from src.models.evaluate import compute_trading_statistics, evaluate_classifier
from src.nlp.preprocessor import TradingTextPreprocessor
from src.nlp.vectorizer import build_tfidf_vectorizer, extract_top_tfidf_features


# ==============================================================================
# 1. PRUEBAS DE QA DEL PREPROCESADOR NLP (TradingTextPreprocessor)
# ==============================================================================

def test_preprocessor_normalization_rsi_levels() -> None:
    """QA: Verifica que los niveles numéricos de RSI se traduzcan en tokens semánticos."""
    prep = TradingTextPreprocessor(mode="lemma")

    # Sobreventa (< 30)
    norm_oversold = prep.normalize("Entré cuando el RSI estaba en 28")
    assert "_nivel_sobreventa_" in norm_oversold, f"Fallo al inyectar sobreventa: {norm_oversold}"
    assert "28" in norm_oversold

    # Sobrecompra (> 70)
    norm_overbought = prep.normalize("Vendí con RSI en 78 en Volatility 100")
    assert "_nivel_sobrecompra_" in norm_overbought, f"Fallo al inyectar sobrecompra: {norm_overbought}"
    assert "volatility_100" in norm_overbought


def test_preprocessor_emojis_translation() -> None:
    """QA: Verifica que los emojis operativos de trading se traduzcan a tokens semánticos."""
    prep = TradingTextPreprocessor(mode="lemma")
    text_win = "Operación ganada con RSI ✅ 🚀"
    norm_win = prep.normalize(text_win)
    assert "_emoji_ganancia_" in norm_win

    text_loss = "Terminé perdiendo ❌ 📉"
    norm_loss = prep.normalize(text_loss)
    assert "_emoji_perdida_" in norm_loss


def test_preprocessor_stop_words_preserves_negations() -> None:
    """
    QA CRÍTICO DE EXAMEN:
    Verifica que las palabras de negación ('no', 'sin', 'nunca', 'antes')
    NUNCA sean eliminadas por el filtro de stop words.
    """
    prep = TradingTextPreprocessor(mode="lemma", keep_negations=True)
    raw_text = "Entré en compra pero no esperé el cierre de la vela y operé sin stop loss"
    norm = prep.normalize(raw_text)
    tokens = prep.tokenize(norm)
    filtered = prep.remove_stopwords(tokens)

    # Conectores vacíos deben eliminarse
    assert "en" not in filtered
    assert "pero" not in filtered
    assert "el" not in filtered
    assert "de" not in filtered
    assert "la" not in filtered
    assert "y" not in filtered

    # Negaciones y condicionales DEBEN conservarse obligatoriamente
    assert "no" in filtered, "¡ERROR GRAVE: La negación 'no' fue eliminada!"
    assert "sin" in filtered, "¡ERROR GRAVE: La preposición privativa 'sin' fue eliminada!"


def test_preprocessor_lemmatization() -> None:
    """QA: Verifica que las conjugaciones en pasado y gerundio se lleven a su lema infinitivo base."""
    prep = TradingTextPreprocessor(mode="lemma")
    tokens = ["entré", "esperé", "perdiendo", "cerré", "gané", "velas"]
    lemmas = prep.lemmatize_tokens(tokens)

    expected = ["entrar", "esperar", "perder", "cerrar", "ganar", "vela"]
    assert lemmas == expected, f"Lematización incorrecta: {lemmas} vs {expected}"


def test_preprocessor_entity_extraction_kenji_case() -> None:
    """QA: Verifica la extracción estructurada sobre el caso acordado con el docente."""
    prep = TradingTextPreprocessor(mode="lemma")
    text = "Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo"
    entities = prep.extract_trading_entities(text)

    assert entities["indicator_code"] == "RSI"
    assert "28" in entities["indicator"]
    assert "Sobreventa" in entities["indicator"]
    assert entities["action"] == "Compra"
    assert entities["operational_error"] == "No esperar cierre de vela"
    assert "Pérdida" in entities["outcome_detected"]


# ==============================================================================
# 2. PRUEBAS DE QA DEL VECTORIZADOR TF-IDF (vectorizer.py)
# ==============================================================================

def test_tfidf_vectorizer_bigrams_builder() -> None:
    """QA: Verifica que el vectorizador genere unigramas y bigramas correctamente."""
    vec = build_tfidf_vectorizer(ngram_range=(1, 2), min_df=1)
    corpus = [
        "entrar compra rsi 28 no esperar cierre vela perder",
        "operar volatility 100 cruce ema confirmacion vela ganar",
    ]
    X = vec.fit_transform(corpus)
    feature_names = list(vec.get_feature_names_out())

    # Bigramas de trading esperados
    assert "no esperar" in feature_names
    assert "cierre vela" in feature_names
    assert "entrar compra" in feature_names
    assert "cruce ema" in feature_names
    assert X.shape[0] == 2
    assert X.shape[1] > 10


def test_extract_top_tfidf_features_production_pipeline() -> None:
    """QA: Verifica que el extractor de pesos activos obtenga los bigramas más pesados."""
    model_path = ROOT_DIR / "models_prod" / "pipeline_prod.joblib"
    if not model_path.exists():
        pytest.skip("Artefacto de producción no disponible.")

    pipeline = joblib.load(model_path)
    processed_text = "entrar compra rsi 28 _nivel_sobreventa_ no esperar cierre vela terminar perder"
    top_features = extract_top_tfidf_features(pipeline, processed_text, top_k=6)

    assert len(top_features) > 0
    terms = [item["term"] for item in top_features]
    assert any("no" in t or "esperar" in t or "rsi" in t for t in terms)
    # Deben estar ordenados descendentemente por peso
    weights = [item["weight"] for item in top_features]
    assert weights == sorted(weights, reverse=True)


# ==============================================================================
# 3. PRUEBAS DE QA DE EVALUACIÓN Y ESTADÍSTICAS (evaluate.py)
# ==============================================================================

def test_evaluate_classifier_metrics_formula() -> None:
    """QA: Verifica el cálculo de Accuracy, Precision, Recall, F1 y error_rate."""
    class DummyModel:
        def predict(self, X: list[str]) -> list[str]:
            return ["Ganadora_Disciplinada", "Perdedora_Error_Operativo", "Riesgo_Neutral"]

    dummy = DummyModel()
    y_true = ["Ganadora_Disciplinada", "Perdedora_Error_Operativo", "Perdedora_Error_Operativo"]
    X_test = ["trade 1", "trade 2", "trade 3"]

    results = evaluate_classifier("Dummy", dummy, X_test, y_true)

    # 2 de 3 acertadas = 0.6667
    assert results["accuracy"] == round(2 / 3, 4)
    assert "confusion_matrix" in results
    assert "per_class" in results
    # Verificar que error_rate esté presente para la modificación en vivo
    assert "error_rate" in results["per_class"]["Perdedora_Error_Operativo"]


def test_compute_trading_statistics_accuracy() -> None:
    """QA: Verifica que las estadísticas históricas agreguen correctamente winrate y errores."""
    sample_records = [
        {
            "id": 1,
            "prediction": "Ganadora_Disciplinada",
            "indicator": "RSI (28)",
            "operational_error": "Ninguno",
            "market": "Volatility 100",
        },
        {
            "id": 2,
            "prediction": "Perdedora_Error_Operativo",
            "indicator": "RSI (28)",
            "operational_error": "No esperar cierre de vela",
            "market": "Volatility 100",
        },
        {
            "id": 3,
            "prediction": "Perdedora_Error_Operativo",
            "indicator": "EMA 20",
            "operational_error": "Operar sin Stop Loss",
            "market": "Crash 500",
        },
    ]

    stats = compute_trading_statistics(sample_records)

    assert stats["total_trades"] == 3
    # 1 win de 3 = 33.3%
    assert stats["global_winrate"] == 33.3
    assert stats["outcome_distribution"]["Ganadora_Disciplinada"] == 1
    assert stats["outcome_distribution"]["Perdedora_Error_Operativo"] == 2
    assert len(stats["operational_errors"]) == 2
    assert stats["volatility_100_summary"]["total_trades"] == 2
    assert "No esperar cierre de vela" in stats["volatility_100_summary"]["top_error"]
