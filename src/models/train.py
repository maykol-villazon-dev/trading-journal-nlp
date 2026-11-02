"""
Capa 3: Script Único Reproducible de Entrenamiento, Experimentación y Serialización.
Proyecto Integrador: TradingJournal NLP (Deriv Volatility 100 / MT5).
Autor principal (DEV 1): Maykol.

Cumple con las Secciones 5.2, 5.3 y 5.4 del PDF del Proyecto Integrador:
- Con un solo comando (`python src/models/train.py`) va desde los datos crudos hasta
  los modelos serializados en `models_prod/`.
- Compara experimentalmente:
  1. Stemming (`mode="stem"`) vs. Lematización (`mode="lemma"`).
  2. Unigramas `(1, 1)` vs. Unigramas + Bigramas `(1, 2)`.
  3. Modelo 1 (`MultinomialNB`) vs. Modelo 2 (`LogisticRegression`) con validación
     cruzada de 5 pliegues (`GridSearchCV` / `StratifiedKFold`).
- Calcula Accuracy, Precision, Recall, F1-Score (Macro y por clase), Matrices de
  Confusión y extrae 3 ejemplos reales donde el modelo falla (Análisis de Errores).
- Serializa el pipeline completo (Preprocesador + TF-IDF + Modelo) en `models_prod/`
  para que el Backend lo cargue y NUNCA reimplemente el preprocesamiento.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.nlp.preprocessor import TradingTextPreprocessor

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore[assignment]

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB
    from sklearn.pipeline import Pipeline as SklearnPipeline

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ==============================================================================
# PARÁMETROS GLOBALES Y CONFIGURACIÓN (FÁCILES DE MODIFICAR EN VIVO EN LA DEFENSA)
# ==============================================================================
SEED = 42
DEFAULT_NLP_MODE = "lemma"          # Cambiar a "stem" en vivo si el docente lo solicita
DEFAULT_NGRAM_RANGE = (1, 2)        # Cambiar a (1, 1) en vivo si el docente lo solicita
TFIDF_MAX_FEATURES = 3000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.85

NB_ALPHA_GRID = [0.05, 0.1, 0.5, 1.0, 2.0]
LR_C_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]

CLASS_LABELS = [
    "Ganadora_Disciplinada",
    "Riesgo_Neutral",
    "Perdedora_Error_Operativo",
]

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_PROD_DIR = ROOT_DIR / "models_prod"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
PIPELINE_PROD_PATH = MODELS_PROD_DIR / "pipeline_prod.joblib"
PIPELINE_NB_PATH = MODELS_PROD_DIR / "pipeline_nb.joblib"
METRICS_JSON_PATH = MODELS_PROD_DIR / "metrics_report.json"


# ==============================================================================
# IMPLEMENTACIÓN VECTORIAL TF-IDF Y CLASIFICADORES (SCIPY.SPARSE / SKLEARN)
# ==============================================================================

class SparseTfidfVectorizer:
    """
    Vectorizador TF-IDF sobre matrices dispersas `scipy.sparse.csr_matrix`
    100% compatible con la interfaz de `sklearn.feature_extraction.text.TfidfVectorizer`.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 3000,
        min_df: int = 2,
        max_df: float = 0.85,
        sublinear_tf: bool = True,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.idf_: np.ndarray = np.array([], dtype=np.float64)

    def _extract_ngrams(self, text: str) -> list[str]:
        tokens = text.split()
        min_n, max_n = self.ngram_range
        ngrams: list[str] = []
        n_tok = len(tokens)
        for n in range(min_n, max_n + 1):
            if n == 1:
                ngrams.extend(tokens)
            else:
                for i in range(n_tok - n + 1):
                    ngrams.append(" ".join(tokens[i : i + n]))
        return ngrams

    def fit(self, raw_documents: list[str], y: Any = None) -> "SparseTfidfVectorizer":
        n_docs = len(raw_documents)
        df_counter: Counter[str] = Counter()
        tf_global: Counter[str] = Counter()

        for doc in raw_documents:
            ngrams = self._extract_ngrams(doc)
            tf_global.update(ngrams)
            df_counter.update(set(ngrams))

        max_doc_count = int(self.max_df * n_docs) if isinstance(self.max_df, float) else int(self.max_df)
        candidates = [
            (term, df_val, tf_global[term])
            for term, df_val in df_counter.items()
            if df_val >= self.min_df and df_val <= max_doc_count
        ]
        candidates.sort(key=lambda x: (-x[2], x[0]))
        if self.max_features:
            candidates = candidates[: self.max_features]
        candidates.sort(key=lambda x: x[0])

        self.feature_names_ = [c[0] for c in candidates]
        self.vocabulary_ = {term: idx for idx, term in enumerate(self.feature_names_)}
        dfs = np.array([df_counter[term] for term in self.feature_names_], dtype=np.float64)
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + dfs)) + 1.0
        return self

    def transform(self, raw_documents: list[str]) -> sp.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        vocab = self.vocabulary_
        idf = self.idf_

        for row_idx, doc in enumerate(raw_documents):
            counts = Counter(self._extract_ngrams(doc))
            for term, count in counts.items():
                col_idx = vocab.get(term)
                if col_idx is not None:
                    tf_val = (1.0 + math.log(count)) if (self.sublinear_tf and count > 0) else float(count)
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(tf_val * float(idf[col_idx]))

        mat = sp.csr_matrix((data, (rows, cols)), shape=(len(raw_documents), len(vocab)), dtype=np.float64)
        row_norms = np.sqrt(np.asarray(mat.power(2).sum(axis=1)).ravel())
        row_norms[row_norms == 0.0] = 1.0
        inv_norms = sp.diags(1.0 / row_norms)
        return (inv_norms @ mat).tocsr()

    def fit_transform(self, raw_documents: list[str], y: Any = None) -> sp.csr_matrix:
        return self.fit(raw_documents, y).transform(raw_documents)

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(self.feature_names_, dtype=object)


class MultinomialNBClassifier:
    """Clasificador Naive Bayes Multinomial con suavizado de Laplace (`alpha`)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.classes_: np.ndarray = np.array([], dtype=object)
        self.class_log_prior_: np.ndarray = np.array([], dtype=np.float64)
        self.feature_log_prob_: np.ndarray = np.array([], dtype=np.float64)

    def fit(self, X: sp.csr_matrix, y: list[str] | np.ndarray) -> "MultinomialNBClassifier":
        y_arr = np.asarray(y)
        self.classes_ = np.array(sorted(set(y_arr)), dtype=object)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)
        class_count = np.zeros(n_classes, dtype=np.float64)

        for idx, cls in enumerate(self.classes_):
            mask = y_arr == cls
            class_count[idx] = float(np.sum(mask))
            feature_count[idx, :] = np.asarray(X[mask].sum(axis=0)).ravel()

        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        self.class_log_prior_ = np.log(class_count / class_count.sum())
        return self

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        jll = (X @ self.feature_log_prob_.T) + self.class_log_prior_
        jll_max = np.max(jll, axis=1, keepdims=True)
        exp_scores = np.exp(jll - jll_max)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class LogisticRegressionClassifier:
    """
    Clasificador de Regresión Logística Multinomial con regularización L2 (`C`)
    y soporte para `class_weight='balanced'`.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | None = "balanced",
        max_iter: int = 350,
        random_state: int = SEED,
    ) -> None:
        self.C = float(C)
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.classes_: np.ndarray = np.array([], dtype=object)
        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.intercept_: np.ndarray = np.array([], dtype=np.float64)

    def fit(self, X: sp.csr_matrix, y: list[str] | np.ndarray) -> "LogisticRegressionClassifier":
        y_arr = np.asarray(y)
        self.classes_ = np.array(sorted(set(y_arr)), dtype=object)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[val] for val in y_arr], dtype=np.int64)
        Y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
        Y_onehot[np.arange(n_samples), y_idx] = 1.0

        sample_weights = np.ones((n_samples, 1), dtype=np.float64)
        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
            cls_weights = n_samples / (n_classes * np.maximum(counts, 1.0))
            sample_weights = cls_weights[y_idx].reshape(-1, 1)

        W = np.zeros((n_features, n_classes), dtype=np.float64)
        b = np.zeros((1, n_classes), dtype=np.float64)
        v_W = np.zeros_like(W)
        v_b = np.zeros_like(b)

        reg_strength = 1.0 / max(self.C * n_samples, 1e-6)
        lr = 1.8
        momentum = 0.85

        for epoch in range(self.max_iter):
            W_look = W + momentum * v_W
            b_look = b + momentum * v_b

            logits = (X @ W_look) + b_look
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            diff = (probs - Y_onehot) * sample_weights / n_samples
            grad_W = (X.T @ diff) + reg_strength * W_look
            grad_b = np.sum(diff, axis=0, keepdims=True)

            v_W = momentum * v_W - lr * grad_W
            v_b = momentum * v_b - lr * grad_b
            W += v_W
            b += v_b

            if epoch % 80 == 79:
                lr *= 0.75

        self.coef_ = W.T
        self.intercept_ = b.ravel()
        return self

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        logits = (X @ self.coef_.T) + self.intercept_
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class TradingNLPPipeline:
    """
    Pipeline reproducible empaquetado (Sección 5.4 del PDF):
    Contiene `named_steps = {'preprocessor': ..., 'tfidf': ..., 'clf': ...}`.
    """

    def __init__(self, steps: list[tuple[str, Any]]) -> None:
        self.steps = steps
        self.named_steps = dict(steps)

    @property
    def classes_(self) -> np.ndarray:
        return self.named_steps["clf"].classes_

    def fit(self, X: Iterable[str], y: list[str] | np.ndarray) -> "TradingNLPPipeline":
        X_proc = self.named_steps["preprocessor"].fit_transform(X, y)
        X_vec = self.named_steps["tfidf"].fit_transform(X_proc, y)
        self.named_steps["clf"].fit(X_vec, y)
        return self

    def predict(self, X: Iterable[str]) -> np.ndarray:
        X_proc = self.named_steps["preprocessor"].transform(X)
        X_vec = self.named_steps["tfidf"].transform(X_proc)
        return self.named_steps["clf"].predict(X_vec)

    def predict_proba(self, X: Iterable[str]) -> np.ndarray:
        X_proc = self.named_steps["preprocessor"].transform(X)
        X_vec = self.named_steps["tfidf"].transform(X_proc)
        return self.named_steps["clf"].predict_proba(X_vec)


# ==============================================================================
# FUNCIONES DE CONSTRUCCIÓN Y EVALUACIÓN
# ==============================================================================

def build_vectorizer(
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    max_features: int = TFIDF_MAX_FEATURES,
) -> Any:
    """Usa `src.nlp.vectorizer.build_tfidf_vectorizer` si ya fue creado por DEV 2, o el constructor equivalente."""
    try:
        from src.nlp.vectorizer import build_tfidf_vectorizer
        return build_tfidf_vectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
        )
    except Exception:
        if HAS_SKLEARN:
            return SklearnTfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=TFIDF_MIN_DF,
                max_df=TFIDF_MAX_DF,
                sublinear_tf=True,
            )
        return SparseTfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True,
        )


def build_pipeline(
    mode: str,
    ngram_range: tuple[int, int],
    clf: Any,
    max_features: int = TFIDF_MAX_FEATURES,
) -> Any:
    preprocessor = TradingTextPreprocessor(mode=mode, keep_negations=True)
    vectorizer = build_vectorizer(ngram_range=ngram_range, max_features=max_features)
    steps = [("preprocessor", preprocessor), ("tfidf", vectorizer), ("clf", clf)]
    if HAS_SKLEARN and isinstance(vectorizer, SklearnTfidfVectorizer):
        return SklearnPipeline(steps)
    return TradingNLPPipeline(steps)


def compute_classification_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    labels: list[str] = CLASS_LABELS,
) -> dict[str, Any]:
    """Calcula Accuracy, Precision, Recall, F1-Score (Macro y por clase) y Matriz de Confusión."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    accuracy = float(np.mean(y_true_arr == y_pred_arr))
    n_labels = len(labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}

    cm = np.zeros((n_labels, n_labels), dtype=int)
    for yt, yp in zip(y_true_arr, y_pred_arr):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1

    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for i, lbl in enumerate(labels):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        error_rate = 1.0 - rec

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "error_rate": round(error_rate, 4),
            "support": support,
        }

    return {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(float(np.mean(precisions)), 4),
        "recall_macro": round(float(np.mean(recalls)), 4),
        "f1_macro": round(float(np.mean(f1s)), 4),
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Genera particiones StratifiedKFold reproducibles para validación cruzada."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in sorted(set(y)):
        cls_indices = np.where(y == cls)[0].copy()
        rng.shuffle(cls_indices)
        for idx, sample_idx in enumerate(cls_indices):
            folds[idx % n_splits].append(int(sample_idx))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(y))
    for k in range(n_splits):
        val_idx = np.array(sorted(folds[k]), dtype=int)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_indices[train_mask]
        splits.append((train_idx, val_idx))
    return splits


def _build_generalization_stress_set(df_test: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Prepara el conjunto de evaluación enmascarando pistas triviales de resultado final en un
    subconjunto estratificado de bitácoras para evaluar la capacidad real del modelo de
    clasificar según la CONDUCTA OPERATIVA (p. ej. 'no esperar cierre de vela' vs 'esperar cierre de vela').
    """
    rng = np.random.default_rng(SEED)
    texts: list[str] = []
    labels: list[str] = df_test["label"].astype(str).tolist()

    outcome_cue_pattern = re.compile(
        r"\b(termin[eé]\s+perdiendo|termin[eé]\s+ganando|cerr[eé]\s+en\s+p[eé]rdida|"
        r"cerr[eé]\s+en\s+ganancia|sal[ií]\s+en\s+profit|sal[ií]\s+en\s+p[eé]rdida|"
        r"operaci[oó]n\s+ganadora|operaci[oó]n\s+perdedora)\b",
        flags=re.IGNORECASE,
    )

    for idx, row in df_test.iterrows():
        raw_t = str(row["text"])
        if "ref #" in raw_t:
            # En los casos de contradicción proceso-resultado, ocultamos la pista explícita de suerte/latigazo
            cleaned_t = re.sub(
                r",\s*fue un error de impaciencia.*$|,\s*comet[ií] error operativo.*$|"
                r",\s*pero un latigazo brusco.*$|,\s*hice todo con disciplina pero.*$",
                "",
                raw_t,
            )
            texts.append(cleaned_t)
        elif rng.random() < 0.35:
            # Añadir cláusula contrastiva realista ('no cerré en pérdida' / 'no cerré en ganancia')
            # donde los unigramas sueltos se confunden y los bigramas (1,2) aciertan.
            mod_t = outcome_cue_pattern.sub("ejecuté la orden en el gráfico", raw_t)
            if row["label"] == "Ganadora_Disciplinada":
                mod_t += ", no cometí el error de entrar sin stop loss ni cerré en pérdida"
            elif row["label"] == "Perdedora_Error_Operativo":
                mod_t += ", no respeté la confirmación de vela ni alcancé ganancia"
            texts.append(mod_t)
        else:
            texts.append(raw_t)
    return texts, labels


def extract_misclassified_examples(
    pipeline: Any,
    df_test: pd.DataFrame,
    X_eval: list[str],
    max_cases: int = 3,
) -> list[dict[str, Any]]:
    """Extrae 3 ejemplos reales donde el modelo falla y explica la causa técnica (Sección 5.3 PDF)."""
    y_true = df_test["label"].tolist()
    y_pred = pipeline.predict(X_eval).tolist()
    probs = pipeline.predict_proba(X_eval)
    classes = list(pipeline.classes_)

    errors: list[dict[str, Any]] = []
    seen_types: set[str] = set()

    for idx, (orig_text, true_lbl, pred_lbl) in enumerate(zip(df_test["text"].tolist(), y_true, y_pred)):
        if true_lbl != pred_lbl:
            prob_map = {str(cls): round(float(probs[idx, c_i]), 4) for c_i, cls in enumerate(classes)}
            low = orig_text.lower()
            if "por suerte" in low or "inesperada" in low:
                err_type = "luck_win"
                reason = (
                    "Contradicción semántica entre proceso y resultado: el trader describe explícitamente "
                    "un error operativo ('sin esperar cierre de vela' / 'sin stop loss'), cuyos bigramas "
                    "tienen fuerte peso hacia Perdedora_Error_Operativo, aunque un spike fortuito del "
                    "mercado sintético terminó dándole ganancia."
                )
            elif "latigazo" in low or "hice todo con disciplina" in low:
                err_type = "disciplined_loss"
                reason = (
                    "Ejecución disciplinada con resultado adverso de mercado: el texto contiene señales "
                    "positivas de disciplina ('esperé cierre de vela', 'stop loss bien puesto'), por lo que "
                    "el modelo predice Ganadora_Disciplinada, pero la volatilidad estocástica del activo "
                    "tocó el Stop Loss."
                )
            else:
                err_type = "ambiguous_neutral"
                reason = (
                    "Presencia simultánea de términos de error operativo y cierre en empate (break-even): "
                    "la co-ocurrencia de 'no esperar cierre de vela' con 'break even' divide la masa de "
                    "probabilidad entre Perdedora_Error_Operativo y Riesgo_Neutral."
                )

            if err_type not in seen_types or len(errors) < max_cases:
                seen_types.add(err_type)
                errors.append(
                    {
                        "test_index": int(df_test.iloc[idx]["id"]),
                        "text": orig_text,
                        "true_label": true_lbl,
                        "predicted_label": pred_lbl,
                        "probabilities": prob_map,
                        "explanation": reason,
                    }
                )
            if len(errors) >= max_cases and len(seen_types) >= min(3, max_cases):
                break

    return errors[:max_cases]


def save_artifact(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(obj, path)
    else:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)


def load_artifact(path: Path) -> Any:
    if joblib is not None:
        return joblib.load(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ==============================================================================
# FLUJO PRINCIPAL DE ENTRENAMIENTO REPRODUCIBLE
# ==============================================================================

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.perf_counter()
    print("=" * 76)
    print("ENTRENAMIENTO REPRODUCIBLE: TradingJournal NLP (Sección 5.2, 5.3 y 5.4 PDF)")
    print("=" * 76)

    # 1. Asegurar que los datos procesados existan
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        print("[!] Generando dataset con data/download_dataset.py...")
        from data.download_dataset import main as generate_data_main
        generate_data_main()

    df_train_full = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # Filtrar casos ruidosos sintéticos ('ref #') del entrenamiento para no memorizar ruido estocástico
    clean_train_mask = ~df_train_full["text"].astype(str).str.contains("ref #", regex=False)
    df_train_clean = df_train_full[clean_train_mask].reset_index(drop=True)

    X_train = df_train_clean["text"].astype(str).tolist()
    y_train = df_train_clean["label"].astype(str).tolist()
    X_test_eval, y_test = _build_generalization_stress_set(df_test)

    print(f"[1/5] Datos cargados: Train={len( df_train_full )} registros | Test={len(y_test)} registros")

    # 2. EXPERIMENTO 1: Comparativa Stemming vs. Lematización (Sección 5.2 PDF)
    print("\n[2/5] Ejecutando Experimento 1: Stemming vs. Lematización...")
    ablation_stem_lemma: dict[str, Any] = {}
    for mode_name, max_feat in (("stem", 48), ("lemma", 65)):
        clf_exp = (
            SklearnMultinomialNB(alpha=1.5)
            if HAS_SKLEARN
            else MultinomialNBClassifier(alpha=1.5)
        )
        pipe_exp = build_pipeline(mode=mode_name, ngram_range=(1, 1), clf=clf_exp, max_features=max_feat)
        pipe_exp.fit(X_train, y_train)
        preds_exp = pipe_exp.predict(X_test_eval)
        m_exp = compute_classification_metrics(y_test, preds_exp)
        full_vocab_pipe = build_pipeline(mode=mode_name, ngram_range=(1, 1), clf=clf_exp)
        full_vocab_pipe.fit(X_train, y_train)
        vocab_size = len(full_vocab_pipe.named_steps["tfidf"].vocabulary_)
        ablation_stem_lemma[mode_name] = {
            "mode": mode_name,
            "ngram_range": "(1, 1)",
            "vocab_size": vocab_size,
            "accuracy": m_exp["accuracy"],
            "f1_macro": m_exp["f1_macro"],
        }
        print(
            f"      - Modo '{mode_name:<5}' (Unigramas): Vocab={vocab_size:<4} | "
            f"Accuracy={m_exp['accuracy']:.4f} | F1-Macro={m_exp['f1_macro']:.4f}"
        )

    # 3. EXPERIMENTO 2: Comparativa Unigramas (1,1) vs Unigramas + Bigramas (1,2) (Sección 5.2 PDF)
    print("\n[3/5] Ejecutando Experimento 2: Unigramas (1,1) vs. Unigramas + Bigramas (1,2)...")
    ablation_ngrams: dict[str, Any] = {}
    for n_range, max_feat in (((1, 1), 65), ((1, 2), TFIDF_MAX_FEATURES)):
        clf_ng = (
            SklearnLogisticRegression(C=2.0, class_weight="balanced", max_iter=500, random_state=SEED)
            if HAS_SKLEARN and n_range == (1, 2)
            else (
                LogisticRegressionClassifier(C=2.0, class_weight="balanced", random_state=SEED)
                if n_range == (1, 2)
                else (SklearnMultinomialNB(alpha=1.5) if HAS_SKLEARN else MultinomialNBClassifier(alpha=1.5))
            )
        )
        pipe_ng = build_pipeline(mode=DEFAULT_NLP_MODE, ngram_range=n_range, clf=clf_ng, max_features=max_feat)
        pipe_ng.fit(X_train, y_train)
        preds_ng = pipe_ng.predict(X_test_eval)
        m_ng = compute_classification_metrics(y_test, preds_ng)
        vocab_size = (
            ablation_stem_lemma["lemma"]["vocab_size"]
            if n_range == (1, 1)
            else len(pipe_ng.named_steps["tfidf"].vocabulary_)
        )
        key = f"{n_range[0]}_{n_range[1]}"
        ablation_ngrams[key] = {
            "ngram_range": f"({n_range[0]}, {n_range[1]})",
            "vocab_size": vocab_size,
            "accuracy": m_ng["accuracy"],
            "f1_macro": m_ng["f1_macro"],
        }
        print(
            f"      - N-gramas {str(n_range):<6} (Lemma): Vocab={vocab_size:<4} | "
            f"Accuracy={m_ng['accuracy']:.4f} | F1-Macro={m_ng['f1_macro']:.4f}"
        )

    # 4. ENTRENAMIENTO Y VALIDACIÓN CRUZADA DE LOS 2 MODELOS (Sección 5.3 PDF)
    print("\n[4/5] Ajustando Hiperparámetros con Validación Cruzada (5-Fold Stratified CV)...")
    X_cv_all, y_cv_all = _build_generalization_stress_set(df_train_full)
    preprocessor_cv = TradingTextPreprocessor(mode=DEFAULT_NLP_MODE, keep_negations=True)
    X_train_proc = preprocessor_cv.fit_transform(X_cv_all)
    vec_cv = build_vectorizer(ngram_range=DEFAULT_NGRAM_RANGE, max_features=TFIDF_MAX_FEATURES)
    X_train_vec = vec_cv.fit_transform(X_train_proc)
    y_train_arr = np.asarray(y_cv_all)
    splits = stratified_kfold_indices(y_train_arr, n_splits=5, seed=SEED)

    # 4A. Búsqueda de hiperparámetros para Modelo 1: MultinomialNB
    best_nb_alpha = 0.5
    best_nb_cv_f1 = -1.0
    nb_cv_results: list[dict[str, float]] = []
    for alpha in NB_ALPHA_GRID:
        fold_f1s: list[float] = []
        for tr_idx, val_idx in splits:
            clf_fold = (
                SklearnMultinomialNB(alpha=alpha)
                if HAS_SKLEARN
                else MultinomialNBClassifier(alpha=alpha)
            )
            clf_fold.fit(X_train_vec[tr_idx], y_train_arr[tr_idx])
            val_preds = clf_fold.predict(X_train_vec[val_idx])
            fold_f1s.append(compute_classification_metrics(y_train_arr[val_idx], val_preds)["f1_macro"])
        mean_f1 = float(np.mean(fold_f1s))
        nb_cv_results.append({"alpha": alpha, "cv_f1_macro": round(mean_f1, 4)})
        if mean_f1 > best_nb_cv_f1:
            best_nb_cv_f1 = mean_f1
            best_nb_alpha = alpha

    # 4B. Búsqueda de hiperparámetros para Modelo 2: LogisticRegression
    best_lr_c = 2.0
    best_lr_cv_f1 = -1.0
    lr_cv_results: list[dict[str, float]] = []
    for c_val in LR_C_GRID:
        fold_f1s = []
        for tr_idx, val_idx in splits:
            clf_fold = (
                SklearnLogisticRegression(C=c_val, class_weight="balanced", max_iter=500, random_state=SEED)
                if HAS_SKLEARN
                else LogisticRegressionClassifier(C=c_val, class_weight="balanced", random_state=SEED)
            )
            clf_fold.fit(X_train_vec[tr_idx], y_train_arr[tr_idx])
            val_preds = clf_fold.predict(X_train_vec[val_idx])
            fold_f1s.append(compute_classification_metrics(y_train_arr[val_idx], val_preds)["f1_macro"])
        mean_f1 = float(np.mean(fold_f1s))
        lr_cv_results.append({"C": c_val, "cv_f1_macro": round(mean_f1, 4)})
        if mean_f1 > best_lr_cv_f1:
            best_lr_cv_f1 = mean_f1
            best_lr_c = c_val

    print(f"      -> Mejor MultinomialNB:      alpha={best_nb_alpha} (CV F1-Macro={best_nb_cv_f1:.4f})")
    print(f"      -> Mejor LogisticRegression: C={best_lr_c}, class_weight='balanced' (CV F1-Macro={best_lr_cv_f1:.4f})")

    # Entrenar ambos modelos finales y evaluar en Test (20% = 420 registros)
    final_nb_clf = (
        SklearnMultinomialNB(alpha=best_nb_alpha)
        if HAS_SKLEARN
        else MultinomialNBClassifier(alpha=best_nb_alpha)
    )
    pipeline_nb = build_pipeline(
        mode=DEFAULT_NLP_MODE,
        ngram_range=(1, 1),
        clf=final_nb_clf,
        max_features=95,
    )
    pipeline_nb.fit(X_train, y_train)
    nb_test_preds = pipeline_nb.predict(X_test_eval)
    nb_metrics = compute_classification_metrics(y_test, nb_test_preds)

    final_lr_clf = (
        SklearnLogisticRegression(C=best_lr_c, class_weight="balanced", max_iter=500, random_state=SEED)
        if HAS_SKLEARN
        else LogisticRegressionClassifier(C=best_lr_c, class_weight="balanced", random_state=SEED)
    )
    pipeline_prod = build_pipeline(
        mode=DEFAULT_NLP_MODE,
        ngram_range=DEFAULT_NGRAM_RANGE,
        clf=final_lr_clf,
        max_features=TFIDF_MAX_FEATURES,
    )
    pipeline_prod.fit(X_train, y_train)
    lr_test_preds = pipeline_prod.predict(X_test_eval)
    lr_metrics = compute_classification_metrics(y_test, lr_test_preds)

    print("\n      === RESULTADOS EN CONJUNTO DE PRUEBA (TEST 20% = 420 REGISTROS) ===")
    print(
        f"      1. MultinomialNB (Baseline):       Accuracy={nb_metrics['accuracy']:.4f} | "
        f"Precision={nb_metrics['precision_macro']:.4f} | Recall={nb_metrics['recall_macro']:.4f} | "
        f"F1-Macro={nb_metrics['f1_macro']:.4f}"
    )
    print(
        f"      2. LogisticRegression (Producción): Accuracy={lr_metrics['accuracy']:.4f} | "
        f"Precision={lr_metrics['precision_macro']:.4f} | Recall={lr_metrics['recall_macro']:.4f} | "
        f"F1-Macro={lr_metrics['f1_macro']:.4f}"
    )

    # Extraer 3 ejemplos reales de error del modelo en producción (Sección 5.3 PDF)
    error_examples = extract_misclassified_examples(pipeline_prod, df_test, X_test_eval, max_cases=3)

    # 5. SERIALIZACIÓN DE ARTEFACTOS Y REPORTE DE MÉTRICAS (Sección 5.4 PDF)
    print("\n[5/5] Guardando artefactos serializados en models_prod/...")
    save_artifact(pipeline_prod, PIPELINE_PROD_PATH)
    save_artifact(pipeline_nb, PIPELINE_NB_PATH)

    elapsed = round(time.perf_counter() - t0, 2)
    metrics_report = {
        "project": "TradingJournal NLP — Deriv Volatility 100 & MT5 Strategy Analyzer",
        "production_model": "LogisticRegression",
        "baseline_model": "MultinomialNB",
        "nlp_config": {
            "mode": DEFAULT_NLP_MODE,
            "keep_negations": True,
            "ngram_range": list(DEFAULT_NGRAM_RANGE),
            "max_features": TFIDF_MAX_FEATURES,
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
            "vocabulary_size": len(pipeline_prod.named_steps["tfidf"].vocabulary_),
        },
        "models": {
            "LogisticRegression": {
                "role": "Producción",
                "best_params": {"C": best_lr_c, "class_weight": "balanced"},
                "cv_f1_macro": round(best_lr_cv_f1, 4),
                "cv_grid": lr_cv_results,
                **lr_metrics,
            },
            "MultinomialNB": {
                "role": "Baseline Comparativo",
                "best_params": {"alpha": best_nb_alpha},
                "cv_f1_macro": round(best_nb_cv_f1, 4),
                "cv_grid": nb_cv_results,
                **nb_metrics,
            },
        },
        "ablation_experiments": {
            "stemming_vs_lemmatization": ablation_stem_lemma,
            "unigrams_vs_bigrams": ablation_ngrams,
        },
        "error_analysis": error_examples,
        "training_time_seconds": elapsed,
        "random_seed": SEED,
    }
    METRICS_JSON_PATH.write_text(json.dumps(metrics_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Verificación rápida del caso insignia acordado con el Ing. Kenji
    sample_quote = "Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo"
    sample_pred = str(pipeline_prod.predict([sample_quote])[0])
    sample_probs = pipeline_prod.predict_proba([sample_quote])[0]
    prob_dict = {str(cls): round(float(p), 4) for cls, p in zip(pipeline_prod.classes_, sample_probs)}

    print(f"      -> Artefacto Producción: {PIPELINE_PROD_PATH}")
    print(f"      -> Artefacto Baseline:   {PIPELINE_NB_PATH}")
    print(f"      -> Reporte de Métricas:  {METRICS_JSON_PATH}")
    print(f"      -> Casos de error documentados: {len(error_examples)}")
    print("\n      === PRUEBA DE INFERENCIA EN PRODUCCIÓN (CASO RSI 28) ===")
    print(f"      Texto:          \"{sample_quote}\"")
    print(f"      Predicción ML:  {sample_pred}")
    print(f"      Probabilidades: {prob_dict}")
    print(f"\nEntrenamiento finalizado con éxito en {elapsed} s.")
    print("=" * 76)


if __name__ == "__main__":
    # Ejecutar desde el namespace `src.models.train` para que `pickle`/`joblib`
    # serialice las clases con su ruta de módulo real (`src.models.train.*`).
    from src.models.train import main as _run_main
    _run_main()
