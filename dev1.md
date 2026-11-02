# Guía Paso a Paso — DEV 1 (Maykol)
**Rol:** Líder de Datos, Análisis Exploratorio (EDA), Preprocesador NLP + Extracción de Entidades de Trading, Entrenamiento de Modelos (`train.py`) e Informe Técnico.  
**Repositorio:** `https://github.com/maykol-villazon-dev/trading-journal-nlp.git`  
**Documento Maestro de Referencia:** [base.md](./base.md)

---

## Reglas de Evaluación Individual que Protege esta Guía (60 Puntos Individuales)
1. **Contribución en el repositorio (20 pts):** Debes tener **commits frecuentes, descriptivos y con sentido propio** (formato *Conventional Commits*). No subas todo en un solo commit gigante.
2. **Explicación de tu parte (15 pts) + Preguntas individuales (10 pts):** En cada tarea de abajo tienes la justificación exacta que te preguntará el Ing. Kenji sobre tus commits.
3. **Modificación en vivo (15 pts):** Al final de este documento tienes la guía rápida de qué línea tocar si el docente te pide un cambio en vivo en NLP o Modelado.

---

## Mapa de Tus Commits (5 Commits Sustanciales en Orden)

| # | Rama sugerida | Archivo(s) a crear/modificar | Mensaje de Commit (Conventional Commit) |
| :--- | :--- | :--- | :--- |
| **Commit 1** | `main` | `.gitignore`, `requirements.txt`, estructura de carpetas, `base.md`, `dev1.md`, `dev2.md` | `chore(setup): initialize project structure, pinned requirements and developer guides` |
| **Commit 2** | `feat/data-eda` | `data/download_dataset.py`, `data/raw/trading_journal_es.csv`, `data/processed/train.csv`, `data/processed/test.csv` | `feat(data): add spanish trading journal dataset generator and stratified 80/20 split` |
| **Commit 3** | `feat/data-eda` | `notebooks/01_eda_experimentos.ipynb` | `feat(eda): add exploratory analysis of classes, missing values, text length and indicators` |
| **Commit 4** | `feat/nlp-preprocessor` | `src/nlp/preprocessor.py` | `feat(nlp): implement TradingTextPreprocessor with negation stopwords, stem/lemma and entity extraction` |
| **Commit 5** | `feat/model-training` | `src/models/train.py`, `models_prod/*`, `docs/informe_tecnico.md` | `feat(models): add reproducible training pipeline comparing MultinomialNB and LogisticRegression` |

---

## Detalle Paso a Paso por Tarea y Commit

### TAREA 1 (Commit 1): Inicialización de Estructura, Dependencias Fijadas y Guías
**Objetivo:** Dejar la estructura oficial de las 5 capas lista en GitHub para que tanto tú como Oscar puedan trabajar en paralelo sin conflictos.

**Archivos incluidos:**
* `.gitignore`
* `requirements.txt` (con versiones fijadas `==` como exige la Sección 5.4 del PDF)
* Carpetas con `__init__.py` / `.gitkeep`: `data/raw/`, `data/processed/`, `notebooks/`, `src/nlp/`, `src/models/`, `api/`, `frontend/`, `models_prod/`, `tests/`, `docs/`
* `base.md`, `dev1.md`, `dev2.md`

**Comandos Git:**
```powershell
git add .gitignore requirements.txt base.md dev1.md dev2.md data/ notebooks/ src/ api/ frontend/ models_prod/ tests/ docs/
git commit -m "chore(setup): initialize project structure, pinned requirements and developer guides"
git push -u origin main
```

---

### TAREA 2 (Commit 2): Dataset de Bitácoras de Trading en Español y Partición Estratificada
**Rama:** `feat/data-eda`  
**Archivo principal:** `data/download_dataset.py`

**¿Qué debe hacer este script?**
1. Generar/descargar de forma reproducible (`random_state=42`) el corpus `data/raw/trading_journal_es.csv` con **2,100 registros** de bitácoras de trading en español orientadas a **Deriv Volatility 100 / 75, Boom 1000, Crash 500, MT5** e indicadores técnicos (`RSI`, `MACD`, `EMA`, `Bandas de Bollinger`, `Soporte/Resistencia`, `Fibonacci`).
2. Cada fila debe tener las columnas:
   * `id`: Identificador único.
   * `text`: Bitácora en español (ej. *"Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo"*).
   * `label`: Una de las 3 clases:
     * `Ganadora_Disciplinada`
     * `Perdedora_Error_Operativo`
     * `Riesgo_Neutral`
   * `indicator`: Indicador principal mencionado (`RSI`, `MACD`, `EMA`, `Bollinger`, `Soporte_Resistencia`, `Price_Action`).
   * `action`: `Compra` o `Venta`.
   * `market`: `Volatility 100`, `Volatility 75`, `Boom 1000`, `Crash 500`, etc.
   * `operational_error`: `No esperar cierre de vela`, `Operar sin Stop Loss`, `Mover Stop Loss`, `FOMO / Entrada anticipada`, `Sobreoperar`, `Contra tendencia` o `Ninguno (Disciplinado)`.
3. Limpiar valores nulos/duplicados y realizar la partición estratificada **80% entrenamiento (`data/processed/train.csv`) / 20% prueba (`data/processed/test.csv`)** con `train_test_split(..., test_size=0.20, stratify=df["label"], random_state=42)`.
4. Crear también `data/processed/journal_history.json` con registros base estructurados para que el endpoint `/api/v1/stats` de Oscar tenga datos estadísticos desde el primer segundo.

**Comandos Git:**
```powershell
git checkout -b feat/data-eda
python data/download_dataset.py
git add data/
git commit -m "feat(data): add spanish trading journal dataset generator and stratified 80/20 split"
git push -u origin feat/data-eda
```

---

### TAREA 3 (Commit 3): Notebook de Análisis Exploratorio y Experimentos (`01_eda_experimentos.ipynb`)
**Rama:** `feat/data-eda`  
**Archivo principal:** `notebooks/01_eda_experimentos.ipynb`

**¿Qué exige la Sección 4, 5.1 y 5.2 del PDF dentro de este Notebook?**
1. **Documentación del Dataset:** Origen, licencia (MIT Educational) y fecha de obtención.
2. **Valores faltantes y duplicados:** Tabla y verificación con `df.isnull().sum()` y duplicados.
3. **Distribución de clases y desbalance:** Gráfico de barras de `Ganadora_Disciplinada`, `Perdedora_Error_Operativo` y `Riesgo_Neutral`, justificando el uso de `stratify=y` y `class_weight='balanced'`.
4. **Longitud de los textos:** Histograma de número de palabras y caracteres por bitácora.
5. **Palabras e indicadores más frecuentes:** Top términos y distribución de errores operativos por indicador técnico (`RSI`, `MACD`, `EMA`).
6. **Demostración Antes / Después de cada etapa del Pipeline de NLP** sobre el ejemplo:
   *"Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo"*.
7. **Experimentos comparativos:**
   * Tabla comparativa **Stemming vs. Lematización**.
   * Tabla comparativa **Unigramas `(1,1)` vs. Unigramas + Bigramas `(1,2)`**.
   * Comparativa **`MultinomialNB` vs. `LogisticRegression`**.

**Comandos Git:**
```powershell
git add notebooks/01_eda_experimentos.ipynb
git commit -m "feat(eda): add exploratory analysis of classes, missing values, text length and indicators"
git push origin feat/data-eda
# Luego fusionar a main (vía Pull Request en GitHub o merge)
```

---

### TAREA 4 (Commit 4): Preprocesador de NLP y Extractor Estructurado (`src/nlp/preprocessor.py`)
**Rama:** `feat/nlp-preprocessor`  
**Archivo principal:** `src/nlp/preprocessor.py`

**Contrato Técnico (Lo que Oscar y `train.py` van a importar de tu archivo):**
Tu clase `TradingTextPreprocessor(BaseEstimator, TransformerMixin)` debe implementar:
1. **`__init__(self, mode="lemma", keep_negations=True)`:**
   * `mode="lemma"` aplica lematización en español (con soporte para SpaCy `es_core_news_sm` + diccionario morfológico de respaldo de trading en español para que nunca falle aunque el entorno no tenga modelo externo) y `mode="stem"` aplica `SnowballStemmer("spanish")`.
2. **`normalize(self, text: str) -> str`:**
   * Pasa a minúsculas, elimina URLs y signos irrelevantes.
   * Detecta niveles numéricos del RSI: si encuentra `rsi` cerca de un número `< 30` (ej. `28`), inyecta el token semántico `_nivel_sobreventa_`; si es `> 70` (ej. `75`), inyecta `_nivel_sobrecompra_`.
   * Traduce emojis de trading (`✅`, `🟢`, `🚀` -> `_emoji_ganancia_`; `❌`, `🔴`, `📉` -> `_emoji_perdida_`).
3. **`tokenize(self, normalized_text: str) -> list[str]`:**
   * Separa en tokens cuidando siglas técnicas (`rsi`, `macd`, `ema`, `v100`, `v75`, `sl`, `tp`, `fomo`, `mt5`) y tokens con guion bajo.
4. **`remove_stopwords(self, tokens: list[str]) -> list[str]`:**
   * Filtra stop words en español pero **conserva obligatoriamente** `{"no", "sin", "ni", "nunca", "antes", "tampoco", "contra", "mas", "menos", "sube", "cae", "baja"}`.
5. **`stem_or_lemmatize(self, tokens: list[str]) -> list[str]`:**
   * Aplica Lematización o Stemming según `self.mode` y normaliza tildes sobre los lemas resultantes (`"entré"` -> `"entrar"`, `"esperé"` -> `"esperar"`, `"perdiendo"` -> `"perder"`).
6. **`explain_steps(self, text: str) -> dict`:**
   * Devuelve el diccionario exacto que el Backend de Oscar mostrará en `POST /api/v1/predict`:
     ```python
     {
         "raw_text": text,
         "normalized": norm_text,
         "tokens": tokens,
         "stopwords_removed": filtered_tokens,
         "lemmatized_or_stemmed": final_tokens,
     }
     ```
7. **`extract_trading_entities(self, text: str) -> dict`:**
   * Extrae la información estructurada prometida al Ing. Kenji:
     ```python
     {
         "indicator": "RSI (Nivel 28 - Sobreventa)",
         "action": "Compra (BUY)",
         "market": "Volatility 100",
         "operational_error": "No esperar cierre de vela",
         "outcome_detected": "Pérdida (Loss)"
     }
     ```

**Comandos Git:**
```powershell
git checkout -b feat/nlp-preprocessor
git add src/nlp/preprocessor.py
git commit -m "feat(nlp): implement TradingTextPreprocessor with negation stopwords, stem/lemma and entity extraction"
git push -u origin feat/nlp-preprocessor
```

---

### TAREA 5 (Commit 5): Entrenamiento Reproducible (`src/models/train.py`), Artefactos e Informe Técnico
**Rama:** `feat/model-training`  
**Archivos principales:** `src/models/train.py`, `models_prod/pipeline_prod.joblib`, `models_prod/pipeline_nb.joblib`, `models_prod/metrics_report.json`, `docs/informe_tecnico.md`

**¿Qué hace `python src/models/train.py` con un solo comando?**
1. Si no existen los CSV en `data/processed/`, ejecuta automáticamente `download_dataset.py`.
2. Corre el **Experimento 1 (Stemming vs. Lematización)** y guarda sus métricas.
3. Corre el **Experimento 2 (Unigramas `(1,1)` vs. Unigramas + Bigramas `(1,2)`)** usando `build_tfidf_vectorizer` de `src/nlp/vectorizer.py` y guarda sus métricas.
4. Entrena y ajusta con `GridSearchCV` (5-fold Stratified CV):
   * **Modelo 1:** `MultinomialNB` (`alpha`: `[0.01, 0.1, 0.5, 1.0, 2.0]`).
   * **Modelo 2:** `LogisticRegression` (`C`: `[0.1, 0.5, 1.0, 2.0, 5.0]`, `class_weight=['balanced', None]`).
5. Usa las funciones de evaluación de `src/models/evaluate.py` para generar todas las métricas, matrices de confusión y los **3 casos de análisis de errores**.
6. Guarda en `models_prod/`:
   * `pipeline_prod.joblib` (Pipeline completo: `TradingTextPreprocessor` + `TfidfVectorizer` + `LogisticRegression`).
   * `pipeline_nb.joblib` (Pipeline con `MultinomialNB`).
   * `metrics_report.json` (Todas las métricas y comparativas listas para `/api/v1/metrics`).

**Comandos Git:**
```powershell
git checkout -b feat/model-training
python src/models/train.py
git add src/models/train.py models_prod/ docs/informe_tecnico.md
git commit -m "feat(models): add reproducible training pipeline comparing MultinomialNB and LogisticRegression"
git push -u origin feat/model-training
```

---

## Guía de Supervivencia para tu Defensa Individual y Modificación en Vivo (15 pts)

Si el Ing. Kenji te pide modificar algo en vivo el día de la defensa:
1. **"Agrega o quita una stop word y muestra cómo cambia":**
   * Abre `src/nlp/preprocessor.py`, busca el conjunto `NEGATION_AND_TRADING_KEEP` y comenta `"no"`, o agrega una palabra en `CUSTOM_EXTRA_STOPWORDS`.
   * Prueba con: `python -c "from src.nlp.preprocessor import TradingTextPreprocessor; p = TradingTextPreprocessor(); print(p.explain_steps('Entré en compra con RSI en 28 pero no esperé el cierre de la vela'))"`
2. **"Cambia Stemming por Lematización y reentrena":**
   * Abre `src/models/train.py`, cambia `DEFAULT_MODE = "lemma"` a `"stem"` y ejecuta `python src/models/train.py`.
3. **"Cambia un hiperparámetro del modelo y reentrena":**
   * Abre `src/models/train.py`, ajusta la grilla de `C` o `alpha` y ejecuta `python src/models/train.py`.
