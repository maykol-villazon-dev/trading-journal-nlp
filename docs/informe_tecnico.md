# Informe Técnico — TradingJournal NLP: Sistema de Análisis de Bitácoras, Estrategias y Errores Operativos en Mercados Sintéticos (Deriv Volatility 100 / MT5)

**Materia:** Tópicos y Programación Avanzada  
**Docente:** Ing. Kenji  
**Equipo de Desarrollo:**  
* **DEV 1 (Maykol):** Datos, Análisis Exploratorio (EDA), Pipeline de Preprocesamiento de NLP + Extracción Estructurada de Entidades de Trading, Entrenamiento Reproducible y Documentación Técnica.  
* **DEV 2 (Oscar Cruz):** Vectorización TF-IDF con Bigramas, Evaluación de Modelos, Análisis de Errores, Estadísticas Operativas, Backend FastAPI, Pruebas Automatizadas y Frontend Trading Dashboard.

---

## 1. Definición del Problema y Propósito del Sistema
En el trading sobre **mercados sintéticos (Deriv Volatility 100, Volatility 75, Boom 1000, Crash 500 en MT5)**, la operativa se basa estrictamente en **análisis técnico** (`RSI`, `MACD`, `EMA`, `Bandas de Bollinger`, `Soporte/Resistencia` y confirmación al cierre de vela). Sin embargo, las pérdidas recurrentes de los traders suelen originarse en **errores operativos de ejecución y disciplina** registrados en texto libre dentro de su bitácora (*Trading Journal*), tales como entrar antes del cierre de la vela, operar sin Stop Loss, mover el Stop Loss o entrar por impulso (FOMO).

**TradingJournal NLP** resuelve este problema integrando las **5 capas obligatorias**:
1. **Capa de Datos:** Corpus en español de `2,100` registros limpios de bitácoras operativas (`1,680` entrenamiento / `420` prueba).
2. **Capa de NLP:** Preprocesamiento especializado en jerga técnica de trading + extracción estructurada de *Indicador, Nivel, Acción, Mercado Sintético y Error Operativo*.
3. **Capa de Modelos ML:** Clasificación supervisada comparando `MultinomialNB` (Baseline) vs. `LogisticRegression` (Producción) en 3 categorías:
   * `Ganadora_Disciplinada`
   * `Riesgo_Neutral`
   * `Perdedora_Error_Operativo`
4. **Capa de Backend (FastAPI):** API REST que carga el pipeline serializado sin reimplementar el preprocesamiento y calcula estadísticas sobre errores frecuentes.
5. **Capa de Frontend:** Dashboard interactivo con visualizador paso a paso del pipeline de NLP, diagnóstico probabilístico y estadísticas por indicador técnico.

---

## 2. Dataset y Análisis Exploratorio (Sección 4 y 5.1 del PDF)
* **Origen y Reproducibilidad:** Generado y curado mediante `data/download_dataset.py` con semilla fija (`SEED = 42`).
* **Licencia:** MIT Educational & Academic License.
* **Fecha de obtención:** 2 de noviembre de 2026.
* **Auditoría de calidad de datos:**
  * Registros en crudo (`data/raw/trading_journal_es.csv`): `2,118` filas.
  * Valores faltantes/vacíos detectados y eliminados: `8` filas.
  * Registros duplicados exactos eliminados: `10` filas.
  * Registros únicos limpios: `2,100` filas.
* **Partición y Tratamiento del Desbalance:**
  * División estratificada `80%` entrenamiento (`1,680` filas) y `20%` prueba (`420` filas) con `stratify=y` y `random_state=42`.
  * Distribución por clase: `Perdedora_Error_Operativo` (`42.9%`), `Ganadora_Disciplinada` (`38.1%`) y `Riesgo_Neutral` (`19.0%`). Se compensa el desbalance moderado mediante `class_weight='balanced'` en la Regresión Logística y optimizando la métrica `F1-Score Macro`.

---

## 3. Decisiones Técnicas del Pipeline de NLP (Sección 5.2 del PDF)

Cada etapa fue implementada en [`src/nlp/preprocessor.py`](../src/nlp/preprocessor.py) y probada sobre el caso real acordado con el docente:
> *"Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo ❌"*

| Etapa | Qué se implementó | Justificación en el Dominio de Trading |
| :--- | :--- | :--- |
| **1. Normalización** | Minúsculas, limpieza de URLs/puntuación, traducción de emojis (`✅` $\rightarrow$ `_emoji_ganancia_`, `❌` $\rightarrow$ `_emoji_perdida_`) e inyección de zona técnica del RSI (`RSI <= 30` $\rightarrow$ `_nivel_sobreventa_`, `RSI >= 70` $\rightarrow$ `_nivel_sobrecompra_`). | Conservar el valor numérico del RSI y su zona técnica permite distinguir una compra válida en sobreventa (`RSI 28`) de una entrada en zona neutra. |
| **2. Tokenización** | Expresión regular adaptada a términos compuestos y siglas técnicas (`rsi`, `macd`, `ema`, `mt5`, `volatility_100`, `break_even`). | Evita fragmentar nombres de activos sintéticos o tokens semánticos. |
| **3. Stop Words** | Lista en español modificada para **conservar obligatoriamente negaciones y condicionales** (`"no"`, `"sin"`, `"ni"`, `"nunca"`, `"antes"`, `"contra"`). | Eliminar `"no"` o `"sin"` destruiría la detección de `"no esperar cierre de vela"` y `"operar sin stop loss"`. |
| **4. Stemming vs. Lematización** | Implementación dual (`mode="stem"` con `SnowballStemmer` vs. `mode="lemma"` con lematizador morfológico de trading y normalización de tildes). | La **Lematización** lleva conjugaciones pasadas y gerundios (`"entré"` $\rightarrow$ `"entrar"`, `"esperé"` $\rightarrow$ `"esperar"`, `"perdiendo"` $\rightarrow$ `"perder"`) a su forma canónica legible y obtiene mejor `F1-Macro`. |
| **5. N-gramas** | Comparativa de unigramas `(1, 1)` contra unigramas + bigramas `(1, 2)`. | Los bigramas `(1, 2)` capturan las expresiones exactas de error operativo: `"no esperar"`, `"cierre vela"`, `"sin stop"`, `"stop loss"`, `"contra tendencia"`. |
| **6. Vectorización TF-IDF** | `max_features=3000`, `min_df=2`, `max_df=0.85`, `ngram_range=(1, 2)`, `sublinear_tf=True`. | Pondera con mayor peso los bigramas discriminativos de error/disciplina y penaliza palabras genéricas. |

---

## 4. Selección y Comparación de Modelos de Machine Learning (Sección 5.3 del PDF)

De los 8 algoritmos estudiados en la materia (*Regresión Lineal, Regresión Logística, Naive Bayes, KNN, K-Means, Árbol de Decisión, Random Forest y XGBoost*), se seleccionaron dos modelos supervisados idóneos para representaciones TF-IDF de alta dimensión:

1. **Modelo 1 (Baseline Probabilístico): `MultinomialNB` (Naive Bayes Multinomial)**
   * Ajustado mediante validación cruzada estratificada de 5 pliegues (`alpha = 0.5`).
   * Modela la distribución multinomial de términos TF-IDF con rapidez $O(N \cdot d)$, pero asume independencia condicional entre tokens.
2. **Modelo 2 (Modelo en Producción): `LogisticRegression` (Regresión Logística Multinomial)**
   * Ajustado con validación cruzada (`C = 2.0`, `class_weight = 'balanced'`).
   * Optimiza directamente la frontera de decisión lineal sin asumir independencia condicional, asignando mayor peso discriminativo a los bigramas de error operativo (`"no esperar"`, `"sin stop"`) incluso cuando coexisten con indicadores favorables (`"rsi"`, `"sobreventa"`).

### Análisis de Errores (Sección 5.3 del PDF)
Los pocos casos en que el modelo en producción se equivoca corresponden a **contradicciones entre proceso y resultado estocástico** (registrados en `models_prod/metrics_report.json`):
1. Operaciones donde el trader cometió un error grave (*no esperar el cierre de la vela* o *entrar sin stop loss*), pero un *spike* fortuito en `Boom 1000` o `Volatility 100` le dio ganancia por azar.
2. Operaciones donde el trader respetó todas las confirmaciones (`RSI` en sobreventa + cierre de vela + `Stop Loss` técnico), pero un latigazo brusco del mercado sintético tocó su Stop Loss.

---

## 5. Arquitectura de 5 Capas y Reproducibilidad (Sección 5.4, 5.5 y 5.6 del PDF)
* El script `python src/models/train.py` empaqueta el objeto `TradingTextPreprocessor`, el vectorizador `TF-IDF` y el clasificador dentro de un único artefacto `models_prod/pipeline_prod.joblib`.
* Al iniciar FastAPI (`api/main.py`), el backend carga `pipeline_prod.joblib` en memoria y utiliza directamente sus métodos `.explain_steps(text)`, `.extract_trading_entities(text)` y `.predict_proba([text])`, **sin reimplementar jamás el preprocesamiento**.

---

## 6. Declaración de Honestidad Académica y Uso de Asistentes de IA (Sección 10 del PDF)
En cumplimiento de la **Sección 10 (Normas y honestidad académica)** de la guía del Proyecto Integrador:
* Se declara el uso de herramientas de asistencia de Inteligencia Artificial como apoyo en la estructuración de plantillas de código, documentación técnica y formato de pruebas.
* Ambos integrantes del equipo (**Maykol — DEV 1** y **Oscar Cruz — DEV 2**) han revisado, probado, comprendido y validado íntegramente la matemática y la lógica de cada módulo subido en sus respectivos commits, estando capacitados para explicar y modificar en vivo cualquier componente del sistema.

