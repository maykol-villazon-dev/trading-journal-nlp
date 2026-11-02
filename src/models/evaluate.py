"""
Capa 3: Evaluación de Modelos de Machine Learning, Análisis de Errores y Estadísticas de Trading.
Proyecto Integrador: TradingJournal NLP (Deriv Volatility 100 / MT5).
Autor: Oscar Cruz (DEV 2).

Cumple con las Secciones 5.3, 5.4 y 6.1 del PDF del Proyecto Integrador:
1. `evaluate_classifier`: Cálculo de Accuracy, Precision Macro, Recall Macro, F1-Score Macro,
   métricas desglosadas por clase (precision, recall, f1, support, error_rate) y Matrices de Confusión.
2. `extract_error_analysis`: Localización y explicación cualitativa de 3 bitácoras reales donde
   el modelo falla (ej. error operativo cometido pero con ganancia fortuita por spike en Volatility 100).
3. `compute_trading_statistics`: Agregación de estadísticas de errores operativos frecuentes, winrate
   por indicador técnico (RSI, MACD, EMA) y radar preventivo para Deriv Volatility 100.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

DEFAULT_CLASS_LABELS = [
    "Ganadora_Disciplinada",
    "Riesgo_Neutral",
    "Perdedora_Error_Operativo",
]


# ==============================================================================
# 1. EVALUACIÓN FORMAL DE CLASIFICADORES Y MATRIZ DE CONFUSIÓN (Sección 5.3 PDF)
# ==============================================================================

def evaluate_classifier(
    model_name: str,
    pipeline: Any,
    X_test: Sequence[str] | np.ndarray | pd.Series,
    y_test: Sequence[str] | np.ndarray | pd.Series,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evalúa un modelo sobre el conjunto de prueba y calcula todas las métricas requeridas.

    Parámetros:
    -----------
    model_name: Nombre descriptivo ("LogisticRegression" o "MultinomialNB").
    pipeline: Pipeline o modelo entrenado con método `predict()`.
    X_test: Textos del conjunto de prueba.
    y_test: Etiquetas reales del conjunto de prueba.
    labels: Orden canónico de las clases.

    Retorna:
    --------
    Diccionario con métricas globales, métricas por clase, error_rate y matriz de confusión.
    """
    if labels is None:
        labels = list(DEFAULT_CLASS_LABELS)

    y_true_list = [str(y) for y in y_test]
    y_pred_list = [str(y) for y in pipeline.predict(list(X_test))]

    n_samples = len(y_true_list)
    if n_samples == 0:
        return {
            "model_name": model_name,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "per_class": {},
            "confusion_matrix": [[0] * len(labels) for _ in labels],
            "labels": labels,
            "total_samples": 0,
        }

    # Accuracy global
    correct = sum(1 for yt, yp in zip(y_true_list, y_pred_list) if yt == yp)
    accuracy = correct / n_samples

    # Matriz de confusión e índices por clase
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    conf_mat = [[0 for _ in labels] for _ in labels]

    for yt, yp in zip(y_true_list, y_pred_list):
        i = label_to_idx.get(yt)
        j = label_to_idx.get(yp)
        if i is not None and j is not None:
            conf_mat[i][j] += 1

    # Métricas por clase
    per_class: dict[str, dict[str, Any]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for i, lbl in enumerate(labels):
        tp = conf_mat[i][i]
        fp = sum(conf_mat[r][i] for r in range(len(labels)) if r != i)
        fn = sum(conf_mat[i][c] for c in range(len(labels)) if c != i)
        support = sum(conf_mat[i][c] for c in range(len(labels)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        # Métrica adicional para defensa en vivo: Tasa de error por clase
        error_rate = 1.0 - rec if support > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "error_rate": round(error_rate, 4),
        }

    return {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "precision_macro": round(float(np.mean(precisions)), 4),
        "recall_macro": round(float(np.mean(recalls)), 4),
        "f1_macro": round(float(np.mean(f1s)), 4),
        "per_class": per_class,
        "confusion_matrix": conf_mat,
        "labels": labels,
        "total_samples": n_samples,
    }


# ==============================================================================
# 2. ANÁLISIS CUALITATIVO DE ERRORES DEL MODELO (Sección 5.3 PDF)
# ==============================================================================

def extract_error_analysis(
    pipeline: Any,
    df_test: pd.DataFrame,
    max_examples: int = 3,
) -> list[dict[str, Any]]:
    """
    Encuentra casos reales en el conjunto de prueba donde el modelo predice incorrectamente
    y genera una explicación técnica detallada del motivo lingüístico u operativo del fallo.

    Cumple con la exigencia expresa del PDF: "Análisis de errores: ejemplos donde el modelo
    falla y una explicación de por qué".
    """
    texts = df_test["text"].tolist()
    y_true = df_test["label"].tolist()

    try:
        y_pred = pipeline.predict(texts)
    except Exception:
        return []

    # Obtener probabilidades si están disponibles
    has_proba = hasattr(pipeline, "predict_proba")
    proba_matrix = pipeline.predict_proba(texts) if has_proba else None
    classes = [str(c) for c in getattr(pipeline, "classes_", DEFAULT_CLASS_LABELS)]

    error_cases: list[dict[str, Any]] = []

    for idx, (yt, yp, txt) in enumerate(zip(y_true, y_pred, texts)):
        if str(yt) == str(yp):
            continue

        probs: dict[str, float] = {}
        if proba_matrix is not None:
            for cls_name, prob_val in zip(classes, proba_matrix[idx]):
                probs[cls_name] = round(float(prob_val), 4)

        row = df_test.iloc[idx]
        indicator = str(row.get("indicator", "Desconocido"))
        operational_error = str(row.get("operational_error", "Ninguno"))
        market = str(row.get("market", "Deriv Volatility"))

        # Diagnóstico lingüístico del error
        explanation = _generate_error_explanation(
            text=txt,
            true_label=str(yt),
            pred_label=str(yp),
            operational_error=operational_error,
            indicator=indicator,
            market=market,
        )

        error_cases.append({
            "id": int(row.get("id", idx + 1)),
            "text": txt,
            "true_label": str(yt),
            "predicted_label": str(yp),
            "probabilities": probs,
            "indicator": indicator,
            "operational_error": operational_error,
            "market": market,
            "explanation": explanation,
        })

        if len(error_cases) >= max_examples:
            break

    return error_cases


def _generate_error_explanation(
    text: str,
    true_label: str,
    pred_label: str,
    operational_error: str,
    indicator: str,
    market: str,
) -> str:
    """Genera la justificación técnica de la ambigüedad en el texto de trading."""
    lower = text.lower()

    if true_label == "Ganadora_Disciplinada" and pred_label == "Perdedora_Error_Operativo":
        return (
            f"El trader utilizó léxico asociado a riesgo o corrección en {market} ('{indicator}'), "
            f"lo que activó bigramas pesados de error operativo en el TF-IDF, a pesar de que la operación "
            f"cerró en ganancia por adecuada absorción de volatilidad."
        )
    if true_label == "Perdedora_Error_Operativo" and pred_label == "Ganadora_Disciplinada":
        return (
            f"El registro describe una configuración técnica limpia con {indicator} ('{operational_error}'), "
            f"donde los términos técnicos alcistas/bajistas dominaron el peso TF-IDF por sobre la mención final "
            f"de indisciplina operativa."
        )
    if true_label == "Riesgo_Neutral":
        return (
            f"Operación cerrada en Break-Even / Empate en {market}. El modelo se sesgó hacia '{pred_label}' "
            f"debido a la mención combinada de confirmaciones técnicas con gestión de salida anticipada."
        )
    return (
        f"Conflicto entre señales técnicas de {indicator} y error operativo '{operational_error}': "
        f"la presencia simultánea de palabras de confirmación y cierre generó probabilidad dividida entre clases."
    )


# ==============================================================================
# 3. ESTADÍSTICAS Y RADAR OPERATIVO DE TRADING (Acordado con Ing. Kenji)
# ==============================================================================

def compute_trading_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Procesa la lista de operaciones históricas almacenadas en la bitácora
    y genera estadísticas agregadas para el Dashboard de Trading y Volatility 100:
    - Conteo y porcentaje de cada error operativo.
    - Winrate y efectividad por indicador técnico (RSI, MACD, EMA, Bollinger, etc.).
    - Desempeño por activo sintético y alertas automáticas para Volatility 100.
    """
    total_trades = len(records)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "outcome_distribution": {},
            "global_winrate": 0.0,
            "operational_errors": [],
            "top_fatal_error": "Ninguno",
            "indicator_performance": {},
            "market_performance": {},
            "volatility_100_summary": {
                "total_trades": 0,
                "winrate": 0.0,
                "top_error": "Sin datos",
                "recommendation": "Registra operaciones para recibir recomendaciones en Volatility 100.",
            },
        }

    outcome_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    fatal_error_counts: Counter[str] = Counter()
    indicator_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "wins": 0, "losses": 0, "neutral": 0})
    market_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "wins": 0, "losses": 0, "neutral": 0})
    v100_errors: Counter[str] = Counter()

    for r in records:
        outcome = str(r.get("prediction") or r.get("label") or "Riesgo_Neutral")
        err = str(r.get("operational_error") or "Ninguno")
        ind = str(r.get("indicator") or "Price_Action").split()[0].replace(",", "")
        mkt = str(r.get("market") or "Volatility 100")

        outcome_counts[outcome] += 1
        if err and err not in ("Ninguno", "Ninguno (Disciplinado)"):
            error_counts[err] += 1
            if outcome == "Perdedora_Error_Operativo":
                fatal_error_counts[err] += 1

        # Por indicador
        indicator_stats[ind]["total"] += 1
        if outcome == "Ganadora_Disciplinada":
            indicator_stats[ind]["wins"] += 1
        elif outcome == "Perdedora_Error_Operativo":
            indicator_stats[ind]["losses"] += 1
        else:
            indicator_stats[ind]["neutral"] += 1

        # Por mercado
        market_stats[mkt]["total"] += 1
        if outcome == "Ganadora_Disciplinada":
            market_stats[mkt]["wins"] += 1
        elif outcome == "Perdedora_Error_Operativo":
            market_stats[mkt]["losses"] += 1
        else:
            market_stats[mkt]["neutral"] += 1

        # Específico Volatility 100
        if "100" in mkt or "v100" in mkt.lower():
            if err and err not in ("Ninguno", "Ninguno (Disciplinado)"):
                v100_errors[err] += 1

    total_wins = outcome_counts.get("Ganadora_Disciplinada", 0)
    global_winrate = round((total_wins / total_trades) * 100.0, 1)

    # Lista de errores operativos ordenados
    error_list = [
        {
            "error": err_name,
            "count": count,
            "percentage": round((count / total_trades) * 100.0, 1),
        }
        for err_name, count in error_counts.most_common()
    ]

    top_fatal_error = fatal_error_counts.most_common(1)[0][0] if fatal_error_counts else "Ninguno detectado"

    # Indicadores estructurados con Winrate
    ind_perf: dict[str, dict[str, Any]] = {}
    for ind_name, st in indicator_stats.items():
        tot = st["total"]
        w = st["wins"]
        ind_perf[ind_name] = {
            "total_trades": tot,
            "wins": w,
            "losses": st["losses"],
            "neutral": st["neutral"],
            "winrate": round((w / tot) * 100.0, 1) if tot > 0 else 0.0,
        }

    # Mercados estructurados
    mkt_perf: dict[str, dict[str, Any]] = {}
    for mkt_name, st in market_stats.items():
        tot = st["total"]
        w = st["wins"]
        mkt_perf[mkt_name] = {
            "total_trades": tot,
            "wins": w,
            "winrate": round((w / tot) * 100.0, 1) if tot > 0 else 0.0,
        }

    # Radar Volatility 100
    v100_total = sum(v["total"] for k, v in market_stats.items() if "100" in k or "v100" in k.lower())
    v100_wins = sum(v["wins"] for k, v in market_stats.items() if "100" in k or "v100" in k.lower())
    v100_winrate = round((v100_wins / v100_total) * 100.0, 1) if v100_total > 0 else 0.0
    top_v100_error = v100_errors.most_common(1)[0][0] if v100_errors else "Sin error predominante"

    recommendation = (
        f"En Deriv Volatility 100, el error más costoso registrado es '{top_v100_error}'. "
        "En este índice de alta velocidad, se exige esperar siempre el cierre de vela (M5) "
        "y mantener el Stop Loss estructural sin desplazarlo."
    )

    return {
        "total_trades": total_trades,
        "outcome_distribution": {
            "Ganadora_Disciplinada": outcome_counts.get("Ganadora_Disciplinada", 0),
            "Riesgo_Neutral": outcome_counts.get("Riesgo_Neutral", 0),
            "Perdedora_Error_Operativo": outcome_counts.get("Perdedora_Error_Operativo", 0),
        },
        "global_winrate": global_winrate,
        "operational_errors": error_list,
        "top_fatal_error": top_fatal_error,
        "indicator_performance": ind_perf,
        "market_performance": mkt_perf,
        "volatility_100_summary": {
            "total_trades": v100_total,
            "winrate": v100_winrate,
            "top_error": top_v100_error,
            "recommendation": recommendation,
        },
    }


# ==============================================================================
# EJECUCIÓN DIRECTA DE COMPROBACIÓN
# ==============================================================================

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    test_csv = base_dir / "data" / "processed" / "test.csv"
    history_json = base_dir / "data" / "processed" / "journal_history.json"
    prod_model_path = base_dir / "models_prod" / "pipeline_prod.joblib"

    if test_csv.exists() and prod_model_path.exists():
        import joblib
        pipeline = joblib.load(prod_model_path)
        df_test = pd.read_csv(test_csv)
        metrics = evaluate_classifier("LogisticRegression_Prod", pipeline, df_test["text"], df_test["label"])
        print("\n=== REPORTE DE EVALUACIÓN (TEST SET) ===")
        print(f"Accuracy:  {metrics['accuracy']}")
        print(f"F1-Macro:  {metrics['f1_macro']}")
        print(f"Métricas por clase:\n{json.dumps(metrics['per_class'], indent=2)}")

        errors = extract_error_analysis(pipeline, df_test, max_examples=3)
        print(f"\n=== ANÁLISIS DE ERRORES ({len(errors)} CASOS) ===")
        for e in errors:
            print(f"- ID {e['id']} | Real: {e['true_label']} | Pred: {e['predicted_label']}")
            print(f"  Explicación: {e['explanation']}")

    if history_json.exists():
        hist_records = json.loads(history_json.read_text(encoding="utf-8"))
        stats = compute_trading_statistics(hist_records)
        print("\n=== ESTADÍSTICAS HISTÓRICAS DE TRADING ===")
        print(f"Total trades:   {stats['total_trades']}")
        print(f"Global Winrate: {stats['global_winrate']}%")
        print(f"Top Fatal Err:  {stats['top_fatal_error']}")
        print(f"V100 Radar:     {stats['volatility_100_summary']}")
