# Registro de Contribuciones y Trazabilidad del Equipo

**Proyecto Integrador:** TradingJournal NLP — Sistema de Análisis de Bitácoras, Estrategias y Errores de Trading  
**Materia:** Tópicos y Programación Avanzada | **Docente:** Ing. Kenji  
**Equipo:** 2 Desarrolladores (Dev 1: Maykol Villazon | Dev 2: Oscar Cruz)  
**Modalidad de Evaluación Individual:** Commits (20 pts) + Explicación (15 pts) + Preguntas individuales (10 pts) + Modificación en vivo (15 pts).

---

## 1. Cumplimiento de la Regla de Oro Individual (Sección 6.1 del PDF)

> *"Los roles orientan el trabajo pero no lo encierran: **cada integrante debe tener al menos un commit significativo en la parte de NLP o machine learning**, porque todos serán preguntados sobre ello."* (PDF Pág. 5)

Ambos desarrolladores cuentan con aportes técnicos sustanciales y verificables en el historial de Git tanto en la etapa de NLP como en la de Machine Learning:

- **Maykol (DEV 1):**
  - **En NLP:** Creó `src/nlp/preprocessor.py` implementando la clase `TradingTextPreprocessor` con las etapas de normalización selectiva, tokenización especializada, conjunto de stop words protegiendo negaciones (`"no"`, `"sin"`, `"ni"`), lematización y el extractor semántico de entidades técnicas (`RSI`, niveles de sobreventa, acción operativa, temporalidad y errores de trading).
  - **En ML:** Creó `src/models/train.py` con el pipeline reproducible de entrenamiento, búsqueda de hiperparámetros con `GridSearchCV` (5-fold Stratified CV) comparando `LogisticRegression` vs `MultinomialNB`, experimentos de ablación (Stemming vs Lematización, Unigramas vs Bigramas) y serialización de artefactos `.joblib`.

- **Oscar Cruz (DEV 2):**
  - **En NLP:** Creó `src/nlp/vectorizer.py` implementando `build_tfidf_vectorizer` (con parámetros optimizados: unigramas y bigramas `(1, 2)`, `sublinear_tf=True`, `max_features=3000`, `min_df=2`, `max_df=0.85`) y el extractor en tiempo real de pesos TF-IDF activos por bitácora `extract_top_tfidf_features`.
  - **En ML:** Creó `src/models/evaluate.py` implementando la función de evaluación con cálculo de Accuracy, Precision Macro, Recall Macro, F1-Score Macro, métricas por clase (`precision`, `recall`, `f1`, `support`, `error_rate`), matrices de confusión, análisis cualitativo de 3 casos reales de error del modelo y el motor de agregación de estadísticas de trading y radar para **Deriv Volatility 100**.

---

## 2. Mapa Completo de Commits y Trazabilidad

| # | Integrante | Rama | Archivos Principales | Mensaje de Commit (Conventional Commit) |
| :- | :--- | :--- | :--- | :--- |
| **1** | Maykol (DEV 1) | `main` | `.gitignore`, `requirements.txt`, guías base | `chore(setup): initialize project structure, pinned requirements and developer guides` |
| **2** | Maykol (DEV 1) | `feat/data-eda` | `data/download_dataset.py`, `train.csv`, `test.csv` | `feat(data): add spanish trading journal dataset generator and stratified 80/20 split` |
| **3** | Maykol (DEV 1) | `feat/data-eda` | `notebooks/01_eda_experimentos.ipynb` | `feat(eda): add exploratory analysis of classes, missing values, text length and indicators` |
| **4** | Maykol (DEV 1) | `feat/nlp-preprocessor` | `src/nlp/preprocessor.py` | `feat(nlp): implement TradingTextPreprocessor with negation stopwords, stem/lemma and entity extraction` |
| **5** | Maykol (DEV 1) | `feat/model-training` | `src/models/train.py`, `models_prod/*`, `docs/informe_tecnico.md` | `feat(models): add reproducible training pipeline comparing MultinomialNB and LogisticRegression` |
| **6** | Oscar Cruz (DEV 2) | `feat/nlp-vectorizer-eval` | `src/nlp/vectorizer.py` | `feat(nlp): add TF-IDF vectorizer builder with bigrams and active weight extractor` |
| **7** | Oscar Cruz (DEV 2) | `feat/nlp-vectorizer-eval` | `src/models/evaluate.py` | `feat(models): implement evaluation metrics, confusion matrix, error analysis and trading stats` |
| **8** | Oscar Cruz (DEV 2) | `feat/backend-api` | `api/schemas.py`, `api/routes.py`, `api/main.py` | `feat(api): build FastAPI endpoints for prediction, NLP pipeline steps, trading stats and metrics` |
| **9** | Oscar Cruz (DEV 2) | `feat/backend-api` | `tests/test_api.py` | `test(api): add automated pytest suite for endpoints and input validation rules` |
| **10** | Oscar Cruz (DEV 2) | `feat/frontend-dashboard` | `frontend/index.html`, `frontend/styles.css`, `frontend/app.js`, `README.md`, `CONTRIBUCIONES.md` | `feat(frontend): build trading journal & Volatility 100 dashboard with step-by-step NLP view` |

---

## 3. División Detallada de Responsabilidades

### DEV 1 — Maykol
- **Capa 1 (Datos):** Generación reproducible del corpus de 2,100 bitácoras de trading en español (`data/raw/trading_journal_es.csv`) y división estratificada 80% train (`1,680` registros) / 20% test (`420` registros) con semilla fija (`random_state=42`).
- **Análisis Exploratorio (EDA):** Notebook interactivo con análisis de nulos, duplicados, tratamiento justificado del desbalance (`stratify=y`, `class_weight='balanced'`), longitud de textos y distribución de indicadores técnicos (`RSI`, `MACD`, `EMA`).
- **Capa 2 (Preprocesador NLP & Extractor):** Pipeline de normalización, tokenización, stop words con negaciones protegidas y extractor estructurado de entidades de trading (`extract_trading_entities`).
- **Capa 3 (Entrenamiento):** Script reproducible `train.py` con validación cruzada estratificada de 5 pliegues y serialización en `models_prod/`.
- **Documentación:** Informe técnico en `docs/informe_tecnico.md` y declaración de uso de IA (Sección 10 del PDF).

### DEV 2 — Oscar Cruz
- **Capa 2 (Vectorización TF-IDF):** Construcción del vectorizador con unigramas y bigramas `(1, 2)` con escala logarítmica sublineal y extractor dinámico de los términos y bigramas más pesados para cada inferencia (`src/nlp/vectorizer.py`).
- **Capa 3 (Evaluación & Estadísticas):** Métricas de clasificación completas, matrices de confusión, tasa de error por clase para la defensa, análisis de 3 casos reales donde el clasificador falla y generador de estadísticas de trading y radar preventivo para Volatility 100 (`src/models/evaluate.py`).
- **Capa 4 (Backend API REST):** Servidor FastAPI con ciclo de vida `lifespan` que carga los artefactos una sola vez en memoria (sin reimplementar preprocesamiento), validaciones Pydantic de entrada (rechazo de textos vacíos, < 3 palabras o > 1000 caracteres) y 4 endpoints REST (`/health`, `/predict`, `/stats`, `/metrics`).
- **Pruebas Automatizadas:** Suite completa con `pytest` en `tests/test_api.py` validando los endpoints, el caso insignia del RSI en 28 sin esperar vela, casos ganadores, casos break-even y reglas de validación HTTP 422.
- **Capa 5 (Frontend Web):** Interfaz Dark Mode estilo MetaTrader 5 / Deriv con 4 botones de ejemplo rápido, tarjetas de extracción estructurada, visualizador en chips del pipeline de NLP paso a paso (resaltando la negación `"no"` conservada en dorado), ranking interactivo de errores operativos y comparativa de modelos ML.
- **Documentación Operativa:** `README.md` exhaustivo y `CONTRIBUCIONES.md`.
