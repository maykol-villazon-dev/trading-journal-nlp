/**
 * Capa 5: Lógica del Cliente Frontend para TradingJournal NLP.
 * Proyecto Integrador: Tópicos y Programación Avanzada (Ing. Kenji).
 * Autor: Oscar Cruz (DEV 2).
 *
 * Responsabilidades:
 * 1. Comunicación con los 4 endpoints de la API FastAPI (/health, /predict, /stats, /metrics).
 * 2. Visualización paso a paso de cada fase del pipeline de NLP (PDF Sección 5.6).
 * 3. Renderizado de diagnósticos ML con colores corporativos (#10b981, #ef4444, #64748b).
 * 4. Actualización reactiva de estadísticas y radar para Deriv Volatility 100.
 */

// URL base de la API FastAPI (Capa 4)
const API_BASE = "http://localhost:8000/api/v1";

// Palabras de negación u omisión que el frontend resalta en dorado
const NEGATION_WORDS = new Set(["no", "sin", "ni", "nunca", "jamas", "tampoco", "antes", "contra"]);

// Frases de prueba predefinidas (Casos acordados con el Ing. Kenji)
const EXAMPLES = {
  caseRSI: "Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo",
  caseV100Win: "Operé Volatility 100 en compra tras cruce de EMA 20 y confirmación de vela alcista con RSI en 32, respeté el take profit y salí en ganancia",
  caseNoSL: "Vendí en Crash 500 cuando el MACD giró a la baja pero entré sin stop loss y moví mi riesgo, cerré en pérdida fuerte",
  caseBreakEven: "Entré en compra en Volatility 75 sobre soporte con bandas de Bollinger estrechas pero el mercado quedó lateral y cerré en break even sin pérdidas",
};

// =============================================================================
// INICIALIZACIÓN AL CARGAR LA PÁGINA
// =============================================================================
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  checkApiHealth();
  loadTradingStats();
  loadModelMetrics();
});

// =============================================================================
// CONFIGURACIÓN DE EVENT LISTENERS
// =============================================================================
function initEventListeners() {
  const textarea = document.getElementById("journalTextInput");
  const form = document.getElementById("journalForm");
  const btnClear = document.getElementById("btnClearText");
  const btnRefreshStats = document.getElementById("btnRefreshStats");
  const btnRefreshMetrics = document.getElementById("btnRefreshMetrics");

  // Botones de casos rápidos
  document.getElementById("btnCaseRSI").addEventListener("click", () => setExampleText(EXAMPLES.caseRSI));
  document.getElementById("btnCaseV100Win").addEventListener("click", () => setExampleText(EXAMPLES.caseV100Win));
  document.getElementById("btnCaseNoSL").addEventListener("click", () => setExampleText(EXAMPLES.caseNoSL));
  document.getElementById("btnCaseBreakEven").addEventListener("click", () => setExampleText(EXAMPLES.caseBreakEven));

  // Contador de caracteres y palabras en tiempo real
  textarea.addEventListener("input", updateCounters);

  // Botón Limpiar
  btnClear.addEventListener("click", () => {
    textarea.value = "";
    updateCounters();
    document.getElementById("validationAlert").classList.add("hidden");
  });

  // Envío del formulario (POST /api/v1/predict)
  form.addEventListener("submit", handleFormSubmit);

  // Botones de recarga
  btnRefreshStats.addEventListener("click", loadTradingStats);
  btnRefreshMetrics.addEventListener("click", loadModelMetrics);
}

function setExampleText(text) {
  const textarea = document.getElementById("journalTextInput");
  textarea.value = text;
  updateCounters();
  document.getElementById("validationAlert").classList.add("hidden");
  textarea.focus();
}

function updateCounters() {
  const textarea = document.getElementById("journalTextInput");
  const text = textarea.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  const chars = textarea.value.length;

  const wordBadge = document.getElementById("wordCountBadge");
  const charBadge = document.getElementById("charCountBadge");

  wordBadge.textContent = `${words} palabra${words === 1 ? "" : "s"} (mínimo 3)`;
  charBadge.textContent = `${chars} / 1000 caracteres`;

  if (words >= 3) {
    wordBadge.className = "counter-badge valid";
  } else {
    wordBadge.className = "counter-badge invalid";
  }

  if (chars > 1000) {
    charBadge.className = "counter-badge invalid";
  } else {
    charBadge.className = "counter-badge info";
  }
}

// =============================================================================
// 1. MONITOREO Y CONEXIÓN (GET /api/v1/health)
// =============================================================================
async function checkApiHealth() {
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const headerModel = document.getElementById("headerModelTag");
  const headerNlp = document.getElementById("headerNlpTag");

  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    statusDot.className = "status-indicator online";
    statusText.textContent = `API Online (${data.production_model})`;
    headerModel.textContent = data.production_model;
    headerNlp.textContent = `${data.nlp_mode === "lemma" ? "Lematización" : "Stemming"} (Unigramas + Bigramas)`;
  } catch (err) {
    statusDot.className = "status-indicator";
    statusText.textContent = "API Desconectada (Puerto 8000)";
    console.warn("No se pudo conectar a la API:", err);
  }
}

// =============================================================================
// 2. INFERENCIA Y DESGLOSE NLP (POST /api/v1/predict)
// =============================================================================
async function handleFormSubmit(e) {
  e.preventDefault();
  const textarea = document.getElementById("journalTextInput");
  const text = textarea.value.trim();
  const alertBox = document.getElementById("validationAlert");
  const spinner = document.getElementById("loadingSpinner");
  const btnAnalyze = document.getElementById("btnAnalyze");

  alertBox.classList.add("hidden");

  // Validación local inmediata
  const words = text.split(/\s+/).filter(Boolean).length;
  if (words < 3) {
    showAlert("El registro de trading debe contener al menos 3 palabras para un análisis significativo.");
    return;
  }
  if (text.length > 1000) {
    showAlert("El texto excede el límite permitido de 1,000 caracteres.");
    return;
  }

  // Activar estado de carga
  btnAnalyze.disabled = true;
  spinner.classList.remove("hidden");

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      const errMsg = Array.isArray(data.detail) ? data.detail[0].msg : (data.detail || "Error en el servidor");
      throw new Error(errMsg);
    }

    // Renderizar resultados en pantalla
    renderPredictionResults(data);
    // Recargar estadísticas para reflejar la nueva operación
    loadTradingStats();

  } catch (err) {
    showAlert(`Error en la inferencia: ${err.message}`);
  } finally {
    btnAnalyze.disabled = false;
    spinner.classList.add("hidden");
  }
}

function showAlert(message) {
  const alertBox = document.getElementById("validationAlert");
  alertBox.textContent = message;
  alertBox.classList.remove("hidden");
}

function renderPredictionResults(data) {
  const resultsSection = document.getElementById("resultsSection");
  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

  const pred = data.prediction;
  const probs = data.probabilities || {};
  const entities = data.extracted_trading_info || {};
  const steps = data.pipeline_steps || {};

  // 1. Tarjeta de Diagnóstico
  const diagnosisCard = document.getElementById("diagnosisCard");
  const diagnosisTag = document.getElementById("diagnosisTag");
  const confidencePercentage = document.getElementById("confidencePercentage");
  const confidenceBarFill = document.getElementById("confidenceBarFill");
  const inferenceTimeBadge = document.getElementById("inferenceTimeBadge");

  inferenceTimeBadge.textContent = `⏱️ ${data.inference_time_ms} ms`;

  let themeClass = "neutral";
  let labelText = "Riesgo Neutral / Break-Even";

  if (pred === "Ganadora_Disciplinada") {
    themeClass = "win";
    labelText = "✅ Ganadora Disciplinada (Éxito Técnico)";
  } else if (pred === "Perdedora_Error_Operativo") {
    themeClass = "loss";
    labelText = "❌ Perdedora por Error Operativo";
  }

  diagnosisCard.className = `panel-card diagnosis-card ${themeClass}`;
  diagnosisTag.className = `diagnosis-status-tag ${themeClass}`;
  diagnosisTag.textContent = labelText;

  const confPct = Math.round((data.confidence || 0) * 100);
  confidencePercentage.textContent = `${confPct}%`;
  confidenceBarFill.style.width = `${confPct}%`;

  // Barras de probabilidades por clase
  const pWin = Math.round((probs["Ganadora_Disciplinada"] || 0) * 100);
  const pNeutral = Math.round((probs["Riesgo_Neutral"] || 0) * 100);
  const pLoss = Math.round((probs["Perdedora_Error_Operativo"] || 0) * 100);

  document.getElementById("probBarWin").style.width = `${pWin}%`;
  document.getElementById("probNumWin").textContent = `${pWin}%`;

  document.getElementById("probBarNeutral").style.width = `${pNeutral}%`;
  document.getElementById("probNumNeutral").textContent = `${pNeutral}%`;

  document.getElementById("probBarLoss").style.width = `${pLoss}%`;
  document.getElementById("probNumLoss").textContent = `${pLoss}%`;

  // 2. Extracción Estructurada de Trading
  document.getElementById("entityIndicator").textContent = entities.indicator || "--";
  document.getElementById("entityAction").textContent = entities.action || "--";
  document.getElementById("entityMarket").textContent = entities.market || "--";
  document.getElementById("entityTimeframe").textContent = entities.timeframe || "--";
  document.getElementById("entityError").textContent = entities.operational_error || "Ninguno";
  document.getElementById("entityOutcome").textContent = entities.outcome_detected || "--";

  // Consejo contextual para Volatility 100
  const tipText = document.getElementById("v100TipText");
  if (entities.indicator_code === "RSI") {
    tipText.textContent = "En Volatility 100, un RSI < 30 no basta: espera la vela de giro de 5 minutos y no persigas el precio sin confirmación.";
  } else if (entities.indicator_code === "EMA") {
    tipText.textContent = "Al operar cruces de EMA en Deriv Volatility, mantén tu Stop Loss detrás del último swing low y no lo muevas.";
  } else {
    tipText.textContent = "Respeta tu plan de trading en Deriv: la mayor causa de pérdidas en sintéticos es mover el Stop Loss en contra.";
  }

  // 3. Pipeline de NLP Paso a Paso
  renderNlpPipelineSteps(steps);
}

function renderNlpPipelineSteps(steps) {
  document.getElementById("stepRawText").textContent = steps.raw_text || "--";
  document.getElementById("stepNormalized").textContent = steps.normalized || "--";
  document.getElementById("pipelineModePill").textContent = `Modo: ${steps.active_mode === "lemma" ? "Lematización" : "Stemming"}`;

  // Tokens
  const tokensContainer = document.getElementById("stepTokens");
  tokensContainer.innerHTML = "";
  (steps.tokens || []).forEach(tok => {
    const chip = document.createElement("span");
    chip.className = "nlp-chip";
    chip.textContent = tok;
    tokensContainer.appendChild(chip);
  });

  // Stop words (Conservadas vs Descartadas)
  const keptContainer = document.getElementById("stepKeptTokens");
  keptContainer.innerHTML = "";
  const keptNegationsSet = new Set(steps.kept_negations || []);

  (steps.stopwords_removed || []).forEach(tok => {
    const chip = document.createElement("span");
    const isNegation = keptNegationsSet.has(tok) || NEGATION_WORDS.has(tok.toLowerCase());
    chip.className = `nlp-chip ${isNegation ? "preserved-negation" : ""}`;
    chip.textContent = isNegation ? `⭐ ${tok} (Conservado)` : tok;
    keptContainer.appendChild(chip);
  });

  const removedContainer = document.getElementById("stepRemovedStopwords");
  removedContainer.innerHTML = "";
  (steps.removed_stopwords_list || []).forEach(tok => {
    const chip = document.createElement("span");
    chip.className = "nlp-chip discarded-sw";
    chip.textContent = tok;
    removedContainer.appendChild(chip);
  });

  // Lemas
  const lemmasContainer = document.getElementById("stepLemmas");
  lemmasContainer.innerHTML = "";
  (steps.lemmatized_or_stemmed || []).forEach(tok => {
    const chip = document.createElement("span");
    chip.className = "nlp-chip";
    chip.textContent = tok;
    lemmasContainer.appendChild(chip);
  });

  // Top TF-IDF Features
  const tfidfContainer = document.getElementById("stepTfidfBars");
  tfidfContainer.innerHTML = "";
  const features = steps.top_tfidf_features || [];

  if (features.length === 0) {
    tfidfContainer.innerHTML = `<span class="text-muted">Sin términos específicos en vocabulario.</span>`;
  } else {
    const maxWeight = Math.max(...features.map(f => f.weight), 0.01);
    features.forEach(f => {
      const pct = Math.min(Math.round((f.weight / maxWeight) * 100), 100);
      const item = document.createElement("div");
      item.className = "tfidf-item";
      item.innerHTML = `
        <div class="tfidf-header">
          <span class="tfidf-term">"${escapeHtml(f.term)}"</span>
          <span class="tfidf-weight">peso: ${f.weight.toFixed(4)}</span>
        </div>
        <div class="tfidf-bar-bg">
          <div class="tfidf-bar-fill" style="width: ${pct}%;"></div>
        </div>
      `;
      tfidfContainer.appendChild(item);
    });
  }
}

// =============================================================================
// 3. ESTADÍSTICAS HISTÓRICAS DE TRADING (GET /api/v1/stats)
// =============================================================================
async function loadTradingStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) return;
    const data = await res.json();

    // Actualizar KPIs
    document.getElementById("kpiTotalTrades").textContent = data.total_trades;
    document.getElementById("kpiWinrate").textContent = `${data.global_winrate}%`;

    const outDist = data.outcome_distribution || {};
    const wins = outDist["Ganadora_Disciplinada"] || 0;
    const losses = outDist["Perdedora_Error_Operativo"] || 0;
    document.getElementById("kpiWinBreakdown").textContent = `${wins} Ganadas • ${losses} Pérdidas`;

    document.getElementById("kpiTopFatalError").textContent = data.top_fatal_error || "Ninguno";

    const v100 = data.volatility_100_summary || {};
    document.getElementById("kpiV100Winrate").textContent = `${v100.winrate || 0}%`;
    document.getElementById("kpiV100Trades").textContent = `${v100.total_trades || 0} trades en V100`;

    // Radar Volatility 100
    document.getElementById("radarRecommendation").textContent = v100.recommendation || "Sin recomendación.";
    document.getElementById("radarTradesCount").textContent = v100.total_trades || 0;
    document.getElementById("radarWinrateVal").textContent = `${v100.winrate || 0}%`;
    document.getElementById("radarTopError").textContent = v100.top_error || "Ninguno";

    // Ranking de Errores Operativos
    renderErrorBars(data.operational_errors || [], data.total_trades || 1);

    // Tabla de Rendimiento por Indicador
    renderIndicatorsTable(data.indicator_performance || {});

  } catch (err) {
    console.warn("Error al cargar /api/v1/stats:", err);
  }
}

function renderErrorBars(errors, totalTrades) {
  const container = document.getElementById("statsErrorBars");
  container.innerHTML = "";

  if (errors.length === 0) {
    container.innerHTML = `<p class="text-muted">No se registran errores operativos.</p>`;
    return;
  }

  const maxCount = Math.max(...errors.map(e => e.count), 1);

  errors.slice(0, 5).forEach(e => {
    const pctBar = Math.round((e.count / maxCount) * 100);
    const item = document.createElement("div");
    item.className = "error-bar-item";
    item.innerHTML = `
      <div class="error-bar-info">
        <span class="error-name">⚠️ ${escapeHtml(e.error)}</span>
        <span class="error-meta">${e.count} veces (${e.percentage}%)</span>
      </div>
      <div class="error-bar-bg">
        <div class="error-bar-fill" style="width: ${pctBar}%;"></div>
      </div>
    `;
    container.appendChild(item);
  });
}

function renderIndicatorsTable(indicatorStats) {
  const tbody = document.querySelector("#indicatorsTable tbody");
  tbody.innerHTML = "";

  const entries = Object.entries(indicatorStats);
  if (entries.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="text-muted">Sin datos de indicadores.</td></tr>`;
    return;
  }

  // Ordenar por total de operaciones
  entries.sort((a, b) => b[1].total_trades - a[1].total_trades);

  entries.forEach(([indName, st]) => {
    const tr = document.createElement("tr");
    const winrateClass = st.winrate >= 50 ? "high" : "low";
    tr.innerHTML = `
      <td><strong>${escapeHtml(indName)}</strong></td>
      <td>${st.total_trades}</td>
      <td class="win-text">${st.wins}</td>
      <td class="loss-text">${st.losses}</td>
      <td><span class="winrate-tag ${winrateClass}">${st.winrate}%</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// =============================================================================
// 4. COMPARATIVA DE MODELOS ML Y ANÁLISIS DE ERRORES (GET /api/v1/metrics)
// =============================================================================
async function loadModelMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) return;
    const data = await res.json();

    renderModelsTable(data.models || {});
    renderAblations(data.ablation_experiments || {});
    renderErrorAnalysis(data.error_analysis || []);

  } catch (err) {
    console.warn("Error al cargar /api/v1/metrics:", err);
  }
}

function renderModelsTable(models) {
  const tbody = document.querySelector("#modelsComparisonTable tbody");
  tbody.innerHTML = "";

  Object.entries(models).forEach(([modelName, m]) => {
    const tr = document.createElement("tr");
    const isProd = m.role === "Producción";
    const roleBadge = isProd
      ? `<span class="badge-nlp" style="background: var(--color-win);">Producción</span>`
      : `<span class="badge-nlp" style="background: var(--color-neutral);">Baseline</span>`;

    const paramsStr = Object.entries(m.best_params || {})
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");

    tr.innerHTML = `
      <td><strong>${escapeHtml(modelName)}</strong></td>
      <td>${roleBadge}</td>
      <td><code>${escapeHtml(paramsStr || "default")}</code></td>
      <td><strong>${(m.cv_f1_macro || 0).toFixed(4)}</strong></td>
      <td>${(m.accuracy || 0).toFixed(4)}</td>
      <td>${(m.precision_macro || 0).toFixed(4)}</td>
      <td>${(m.recall_macro || 0).toFixed(4)}</td>
      <td><strong style="color: ${isProd ? "var(--color-win)" : "inherit"};">${(m.f1_macro || 0).toFixed(4)}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAblations(ablations) {
  // 1. Stemming vs Lemma
  const stemLemmaBox = document.getElementById("ablationStemLemmaBox");
  stemLemmaBox.innerHTML = "";
  const sl = ablations.stemming_vs_lemmatization || {};

  const lemmaF1 = sl.lemmatization_f1_macro || 0;
  const stemF1 = sl.stemming_f1_macro || 0;

  stemLemmaBox.innerHTML = `
    <div class="ablation-row ${lemmaF1 >= stemF1 ? "winner" : ""}">
      <span>Lematización (SpaCy / Diccionario Canónico):</span>
      <span class="ablation-val">${lemmaF1.toFixed(4)} F1</span>
    </div>
    <div class="ablation-row ${stemF1 > lemmaF1 ? "winner" : ""}">
      <span>Stemming (SnowballStemmer):</span>
      <span class="ablation-val">${stemF1.toFixed(4)} F1</span>
    </div>
    <p class="step-note">
      ${sl.conclusion || "La lematización preserva mejor las conjugaciones del español ('esperé' -> 'esperar')."}
    </p>
  `;

  // 2. Unigramas vs Bigramas
  const ngramsBox = document.getElementById("ablationNgramsBox");
  ngramsBox.innerHTML = "";
  const ng = ablations.unigrams_vs_bigrams || {};

  const biF1 = ng.bigrams_f1_macro || 0;
  const uniF1 = ng.unigrams_f1_macro || 0;

  ngramsBox.innerHTML = `
    <div class="ablation-row ${biF1 >= uniF1 ? "winner" : ""}">
      <span>Unigramas + Bigramas (1, 2) [Producción]:</span>
      <span class="ablation-val">${biF1.toFixed(4)} F1</span>
    </div>
    <div class="ablation-row ${uniF1 > biF1 ? "winner" : ""}">
      <span>Solo Unigramas (1, 1):</span>
      <span class="ablation-val">${uniF1.toFixed(4)} F1</span>
    </div>
    <p class="step-note">
      ${ng.conclusion || "Los bigramas ('no esperar', 'cierre vela') capturan la estructura exacta de las fallas operativas."}
    </p>
  `;
}

function renderErrorAnalysis(errors) {
  const container = document.getElementById("errorAnalysisList");
  container.innerHTML = "";

  if (errors.length === 0) {
    container.innerHTML = `<p class="text-muted">Sin casos de error documentados.</p>`;
    return;
  }

  errors.forEach((err, idx) => {
    const card = document.createElement("div");
    card.className = "error-case-card";
    card.innerHTML = `
      <div class="error-case-header">
        <strong>Caso de Prueba #${err.id || idx + 1} (${escapeHtml(err.market || "Volatility")})</strong>
        <div class="error-case-labels">
          <span>Real: <strong class="win-text">${escapeHtml(err.true_label)}</strong></span>
          <span>Predicha: <strong class="loss-text">${escapeHtml(err.predicted_label)}</strong></span>
        </div>
      </div>
      <div class="error-case-text">"${escapeHtml(err.text)}"</div>
      <div class="error-case-explanation">
        <strong>Diagnóstico Técnico del Modelo:</strong> ${escapeHtml(err.explanation)}
      </div>
    `;
    container.appendChild(card);
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
