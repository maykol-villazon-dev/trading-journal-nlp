# Guía Paso a Paso — DEV 2 (Oscar Cruz)
**Rol:** Especialista en Vectorización TF-IDF, Evaluación y Estadísticas de Errores de Trading (`src/`), Backend FastAPI (`api/`), Pruebas Automatizadas (`tests/`), Frontend Dashboard Volatility 100 (`frontend/`) y Documentación Operativa (`README.md`, `CONTRIBUCIONES.md`).  
**Repositorio:** `https://github.com/maykol-villazon-dev/trading-journal-nlp.git`  
**Documento Maestro de Referencia:** [base.md](./base.md)

---

## ¡IMPORTANTE! Cómo Proteger tus 60 Puntos Individuales (Reglas del PDF del Ing. Kenji)

1. **Regla obligatoria de la Sección 6.1 del PDF (Página 5):**
   > *"Los roles orientan el trabajo pero no lo encierran: **cada integrante debe tener al menos un commit significativo en la parte de NLP o machine learning**, porque todos serán preguntados sobre ello."*
   * Por eso, tus **Tareas 1 y 2** están dentro de `src/nlp/vectorizer.py` y `src/models/evaluate.py`. **No te saltes esos dos commits**, porque son tu respaldo directo de autoría en NLP y Machine Learning ante el docente.
2. **Configura tu cuenta Git antes de hacer tu primer commit:**
   ```powershell
   git clone https://github.com/maykol-villazon-dev/trading-journal-nlp.git
   cd trading-journal-nlp
   git config user.name "Oscar Cruz"
   git config user.email "TU_CORREO_DE_GITHUB@ejemplo.com"
   ```
3. **Haz 1 commit por cada Tarea listada abajo** (no subas todo junto en un solo commit al final).

---

## Mapa de Tus Commits (5 Commits Sustanciales en Orden)

| # | Rama sugerida | Archivo(s) a crear/modificar | Mensaje de Commit (Conventional Commit) |
| :--- | :--- | :--- | :--- |
| **Commit 1** | `feat/nlp-vectorizer-eval` | `src/nlp/vectorizer.py` | `feat(nlp): add TF-IDF vectorizer builder with bigrams and active weight extractor` |
| **Commit 2** | `feat/nlp-vectorizer-eval` | `src/models/evaluate.py` | `feat(models): implement evaluation metrics, confusion matrix, error analysis and trading stats` |
| **Commit 3** | `feat/backend-api` | `api/schemas.py`, `api/routes.py`, `api/main.py` | `feat(api): build FastAPI endpoints for prediction, NLP pipeline steps, trading stats and metrics` |
| **Commit 4** | `feat/backend-api` | `tests/test_api.py` | `test(api): add automated pytest suite for endpoints and input validation rules` |
| **Commit 5** | `feat/frontend-dashboard` | `frontend/index.html`, `frontend/styles.css`, `frontend/app.js`, `README.md`, `CONTRIBUCIONES.md` | `feat(frontend): build trading journal & Volatility 100 dashboard with step-by-step NLP view` |

---

## Detalle Paso a Paso por Tarea y Commit

### TAREA 1 (Commit 1 - Tu aporte en NLP): Vectorizador TF-IDF y Extractor de Pesos (`src/nlp/vectorizer.py`)
**Rama:** `feat/nlp-vectorizer-eval`  
**Archivo a implementar:** `src/nlp/vectorizer.py`

**¿Qué hace este módulo y cómo lo explicarás en la defensa?**
1. **`build_tfidf_vectorizer(ngram_range=(1, 2), max_features=3000, min_df=2, max_df=0.85) -> TfidfVectorizer`:**
   * Construye el vectorizador TF-IDF con parámetros documentados (Sección 5.2 del PDF):
     * `max_features=3000`: Limita el vocabulario a los 3,000 términos más relevantes.
     * `min_df=2`: Descarta palabras que aparecen en 1 sola bitácora (evita errores de tipeo).
     * `max_df=0.85`: Descarta palabras que aparecen en más del 85% de los textos.
     * `ngram_range=(1, 2)`: Incluye **unigramas y bigramas** para capturar frases críticas de trading como `"no esperar"`, `"cierre vela"`, `"stop loss"`, `"entrar compra"`.
     * `sublinear_tf=True`: Aplica escala logarítmica $1 + \log(\text{tf})$ para amortiguar repeticiones.
2. **`extract_top_tfidf_features(pipeline, processed_text: str, top_k: int = 8) -> list[dict]`:**
   * Dado el pipeline entrenado y el texto lematizado de una bitácora (ej. *"entrar compra rsi 28 no esperar cierre vela perder"*), devuelve los términos/bigramas activos con su peso TF-IDF ordenados de mayor a menor:
     ```json
     [
       {"term": "no esperar", "weight": 0.462},
       {"term": "cierre vela", "weight": 0.418},
       {"term": "rsi", "weight": 0.312}
     ]
     ```

**Comandos Git para tu Commit 1:**
```powershell
git checkout -b feat/nlp-vectorizer-eval
git add src/nlp/vectorizer.py
git commit -m "feat(nlp): add TF-IDF vectorizer builder with bigrams and active weight extractor"
git push -u origin feat/nlp-vectorizer-eval
```

---

### TAREA 2 (Commit 2 - Tu aporte en ML): Evaluación de Modelos, Análisis de Errores y Estadísticas de Trading (`src/models/evaluate.py`)
**Rama:** `feat/nlp-vectorizer-eval`  
**Archivo a implementar:** `src/models/evaluate.py`

**¿Qué hace este módulo y cómo cumple lo que acordaste con el Ing. Kenji?**
1. **`evaluate_classifier(model_name: str, pipeline, X_test, y_test, labels: list[str]) -> dict`:**
   * Calcula las métricas exigidas en la Sección 5.3 del PDF para `LogisticRegression` y `MultinomialNB`:
     * `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`
     * Métricas desglosadas por cada clase (`Ganadora_Disciplinada`, `Riesgo_Neutral`, `Perdedora_Error_Operativo`)
     * `confusion_matrix` (matriz 3x3 en lista para pintarla en el Frontend).
2. **`extract_error_analysis(pipeline, df_test, max_examples: int = 3) -> list[dict]`:**
   * Cumple la exigencia de la Sección 5.3: *"Análisis de errores: ejemplos donde el modelo falla y una explicación de por qué"*.
   * Encuentra 3 bitácoras reales del conjunto de prueba donde `y_pred != y_true`, extrae sus probabilidades y genera la explicación técnica del fallo (ej. cuando un trader no esperó el cierre de vela pero aun así ganó por un movimiento brusco de *Volatility 100*).
3. **`compute_trading_statistics(records: list[dict]) -> dict`:**
   * Cumple exactamente lo acordado por WhatsApp con el Ing. Kenji:
     * Cuenta la frecuencia de cada **error operativo** (*No esperar cierre de vela, Operar sin Stop Loss, FOMO, Mover Stop Loss, Contra tendencia*).
     * Calcula el **Winrate (porcentaje de acierto) por indicador técnico** (`RSI`, `MACD`, `EMA`, `Bollinger`, `Soporte/Resistencia`).
     * Calcula el desempeño por activo sintético (`Volatility 100`, `Volatility 75`, `Boom 1000`, `Crash 500`) y genera recomendaciones operativas automáticas.

**Comandos Git para tu Commit 2:**
```powershell
git add src/models/evaluate.py
git commit -m "feat(models): implement evaluation metrics, confusion matrix, error analysis and trading stats"
git push origin feat/nlp-vectorizer-eval
```

---

### TAREA 3 (Commit 3): Backend API REST con FastAPI (`api/schemas.py`, `api/routes.py`, `api/main.py`)
**Rama:** `feat/backend-api`  
**Archivos a implementar:**
1. **`api/schemas.py`:**
   * Modelos Pydantic v2 para request y response.
   * **Validaciones obligatorias de entrada (Sección 5.5 del PDF):**
     * Rechazar texto vacío o solo espacios.
     * Rechazar textos con **menos de 3 palabras** (`"El registro de trading debe contener al menos 3 palabras"`).
     * Rechazar textos con **más de 1,000 caracteres**.
2. **`api/main.py`:**
   * Inicializa FastAPI con CORS habilitado y `lifespan` que carga **una sola vez en memoria** `models_prod/pipeline_prod.joblib`, `models_prod/metrics_report.json` y el historial `data/processed/journal_history.json`.
   * **Regla de Oro del PDF (Sección 5.4):** El backend **nunca reimplementa** el preprocesamiento; usa directamente `pipeline.named_steps["preprocessor"].explain_steps(text)` y `extract_trading_entities(text)` del objeto cargado.
3. **`api/routes.py`:**
   * `GET /api/v1/health`: Estado del servidor, versión y modelo en producción.
   * `POST /api/v1/predict`:
     * Recibe `{"text": "Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo"}`
     * Devuelve:
       * `prediction`: `"Perdedora_Error_Operativo"`
       * `probabilities`: `{"Ganadora_Disciplinada": 0.04, "Riesgo_Neutral": 0.05, "Perdedora_Error_Operativo": 0.91}`
       * `pipeline_steps`: Antes y después de Normalización, Tokenización, Stop words (con `"no"` conservado), Lematización y Top TF-IDF features.
       * `extracted_trading_info`: Indicador (`RSI (Nivel 28 - Sobreventa)`), Acción (`Compra`), Mercado (`Volatility 100`), Error Operativo (`No esperar cierre de vela`) y Resultado.
       * `inference_time_ms`: Tiempo de respuesta en milisegundos.
     * Guarda automáticamente el nuevo registro estructurado en el historial para actualizar las estadísticas.
   * `GET /api/v1/stats`: Devuelve las estadísticas actualizadas de errores frecuentes, winrate por indicador y recomendaciones para **Volatility 100**.
   * `GET /api/v1/metrics`: Devuelve la comparativa `LogisticRegression` vs `MultinomialNB`, matrices de confusión, Stemming vs Lematización, N-gramas y análisis de errores.

**Comandos Git para tu Commit 3:**
```powershell
git checkout -b feat/backend-api
git add api/
git commit -m "feat(api): build FastAPI endpoints for prediction, NLP pipeline steps, trading stats and metrics"
git push -u origin feat/backend-api
```

---

### TAREA 4 (Commit 4): Pruebas Automatizadas del Backend y Pipeline (`tests/test_api.py`)
**Rama:** `feat/backend-api`  
**Archivo a implementar:** `tests/test_api.py`

**¿Qué prueba `pytest tests/ -v`?**
1. `test_health_endpoint()`: Verifica HTTP 200 y modelo cargado.
2. `test_predict_oscar_rsi_example()`: Envía exactamente *"Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo"* y comprueba que detecta `RSI`, acción `Compra`, conserva `"no"` en `stopwords_removed`, identifica el error `No esperar cierre de vela` y predice `Perdedora_Error_Operativo`.
3. `test_predict_disciplined_win()`: Envía una entrada disciplinada en `Volatility 100` y comprueba predicción `Ganadora_Disciplinada`.
4. `test_validation_empty_and_short_text()`: Verifica que textos vacíos o de 1-2 palabras devuelvan HTTP `422`/`400` con mensaje claro.
5. `test_stats_and_metrics_endpoints()`: Verifica que `/api/v1/stats` y `/api/v1/metrics` devuelvan las estructuras completas.

**Comandos Git para tu Commit 4:**
```powershell
pytest tests/ -v
git add tests/test_api.py
git commit -m "test(api): add automated pytest suite for endpoints and input validation rules"
git push origin feat/backend-api
```

---

### TAREA 5 (Commit 5): Frontend Trading Journal & Volatility 100 Dashboard + Documentación (`frontend/`, `README.md`, `CONTRIBUCIONES.md`)
**Rama:** `feat/frontend-dashboard`  
**Archivos a implementar:** `frontend/index.html`, `frontend/styles.css`, `frontend/app.js`, `README.md`, `CONTRIBUCIONES.md`

**¿Qué tiene el Frontend (`frontend/`)?**
1. **Diseño Dark Mode estilo Deriv / MT5 / TradingView:**
   * Verde `#10b981` (`Ganadora_Disciplinada`), Rojo `#ef4444` (`Perdedora_Error_Operativo`), Ámbar/Azul `#64748b` (`Riesgo_Neutral`).
2. **Caja de Registro de Bitácora con Botones de Ejemplo:**
   * Botón 1 (Caso RSI 28 sin esperar vela): *"Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo"*
   * Botón 2 (Caso Ganador Volatility 100): *"Operé Volatility 100 en compra tras cruce de EMA 20 y confirmación de vela alcista con RSI en 32, respeté el take profit y salí en ganancia"*
   * Botón 3 (Caso Sin Stop Loss): *"Vendí en Crash 500 cuando el MACD giró a la baja pero entré sin stop loss y moví mi riesgo, cerré en pérdida fuerte"*
   * Botón 4 (Caso Neutral / Break-Even): *"Entré en compra en Volatility 75 sobre soporte con bandas de Bollinger estrechas pero el mercado quedó lateral y cerré en break even"*
3. **Tarjetas de Extracción Estructurada:** Muestra al instante el **Indicador Técnico**, **Acción**, **Activo Sintético**, **Error Operativo** y **Diagnóstico ML con barra de probabilidad**.
4. **Visualizador del Pipeline de NLP Paso a Paso (Exigencia Sección 5.6 del PDF):**
   * Muestra el Antes y Después en chips: *Texto Original -> Normalizado -> Tokens -> Stop Words eliminadas (destacando las negaciones conservadas como `"no"`) -> Lemas/Raíces -> Top Bigramas TF-IDF*.
5. **Panel de Estadísticas de Trading & Volatility 100:**
   * Ranking de errores operativos más frecuentes del trader, efectividad por indicador (`RSI`, `MACD`, `EMA`) y recomendación algorítmica.
6. **Panel Comparativo de Modelos ML:**
   * Tabla comparativa `LogisticRegression` vs `MultinomialNB`, comparativa Stemming vs Lematización y N-gramas, Matrices de Confusión visuales y los 3 casos de Análisis de Errores.

**Comandos Git para tu Commit 5:**
```powershell
git checkout -b feat/frontend-dashboard
git add frontend/ README.md CONTRIBUCIONES.md
git commit -m "feat(frontend): build trading journal & Volatility 100 dashboard with step-by-step NLP view"
git push -u origin feat/frontend-dashboard
```

---

## Guía Rápida para tu Modificación en Vivo el Día de la Defensa (15 pts)

| Qué te puede pedir el Ing. Kenji | Dónde cambiarlo en 30 segundos |
| :--- | :--- |
| **"En el Backend, cambia la validación para rechazar textos de menos de 5 palabras"** | En `api/schemas.py`, cambia `MIN_WORDS = 3` a `MIN_WORDS = 5`. Guarda y prueba en el navegador o en `http://localhost:8000/docs`. |
| **"En NLP, cambia el rango de n-gramas de `(1,2)` a `(1,1)`"** | En `src/nlp/vectorizer.py`, cambia `ngram_range=(1, 2)` a `ngram_range=(1, 1)` y corre `python src/models/train.py`. |
| **"En Evaluación, agrega una métrica adicional por categoría"** | En `src/models/evaluate.py`, muestra la tasa de error por clase (`error_rate = 1.0 - recall`) y corre `python src/models/train.py`. |
| **"En el Frontend, muestra un dato nuevo que devuelva la API"** | En `frontend/app.js`, muestra en la tarjeta la cantidad de stop words eliminadas o el margen de diferencia entre la 1.ª y 2.ª probabilidad. |
