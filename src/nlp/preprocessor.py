"""
Capa 2: Pipeline de Preprocesamiento de NLP y Extracción Estructurada para Bitácoras de Trading.
Proyecto Integrador: TradingJournal NLP (Deriv Volatility 100 / MT5).
Autor principal (DEV 1): Maykol.

Cumple con las Secciones 5.2 y 5.4 del PDF del Proyecto Integrador:
1. Normalización (minúsculas, tildes, puntuación, números/niveles técnicos de RSI, URLs y emojis).
2. Tokenización adaptada al dominio de trading e índices sintéticos.
3. Stop words en español conservando obligatoriamente las negaciones ("no", "sin", "ni", "antes", "contra").
4. Soporte dual y comparativo para Stemming (`mode="stem"`) y Lematización (`mode="lemma"`).
5. Método `explain_steps(text)` y `extract_trading_entities(text)` para que el Backend
   cargue este mismo objeto serializado y NUNCA reimplemente el preprocesamiento.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    class BaseEstimator:  # type: ignore[no-redef]
        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        def set_params(self, **params: Any) -> "BaseEstimator":
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class TransformerMixin:  # type: ignore[no-redef]
        def fit_transform(self, X: Iterable[str], y: Any = None, **fit_params: Any) -> list[str]:
            return self.fit(X, y, **fit_params).transform(X)

try:
    from nltk.stem import SnowballStemmer
    _NLTK_STEMMER = SnowballStemmer("spanish")
except Exception:
    _NLTK_STEMMER = None


# ==============================================================================
# CONFIGURACIÓN DE STOP WORDS Y REGLAS DE DOMINIO (FÁCIL DE MODIFICAR EN VIVO)
# ==============================================================================

# Palabras de negación y polaridad operativa que NUNCA deben eliminarse como stop words.
# (Para la modificación en vivo del docente: aquí se puede agregar o quitar "no" o "sin").
NEGATION_AND_TRADING_KEEP: set[str] = {
    "no",
    "sin",
    "ni",
    "nunca",
    "jamas",
    "tampoco",
    "antes",
    "contra",
    "mas",
    "menos",
    "sube",
    "cae",
    "baja",
    "si",
    "sí",
}

# Stop words adicionales de dominio que se pueden agregar en vivo si el docente lo solicita
CUSTOM_EXTRA_STOPWORDS: set[str] = set()

# Lista base de stop words del español (conectores, artículos, preposiciones y auxiliares vacíos)
SPANISH_BASE_STOPWORDS: set[str] = {
    "a", "al", "algo", "algunas", "algunos", "ante", "como", "con", "cual", "cuando",
    "de", "del", "desde", "donde", "durante", "e", "el", "ella", "ellas", "ellos",
    "en", "era", "erais", "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso",
    "esos", "esta", "estaba", "estabais", "estaban", "estabas", "estad", "estada",
    "estadas", "estado", "estados", "estamos", "estando", "estar", "estaremos",
    "estará", "estarán", "estarás", "estaré", "estaréis", "estaría", "estaríais",
    "estaríamos", "estarían", "estarías", "estas", "este", "estemos", "esto", "estos",
    "estoy", "estuve", "estuviera", "estuvierais", "estuvieran", "estuvieras",
    "estuvieron", "estuviese", "estuvieseis", "estuviesen", "estuvieses", "estuvimos",
    "estuviste", "estuvisteis", "estuvo", "fue", "fuera", "fuerais", "fueran",
    "fueras", "fueron", "fuese", "fueseis", "fuesen", "fueses", "fui", "fuimos",
    "fuiste", "fuisteis", "ha", "habida", "habidas", "habido", "habidos", "habiendo",
    "habremos", "habrá", "habrán", "habrás", "habré", "habréis", "habría", "habríais",
    "habríamos", "habrían", "habrías", "habéis", "había", "habíais", "habíamos",
    "habían", "habías", "han", "has", "hasta", "hay", "haya", "hayamos", "hayan",
    "hayas", "hayáis", "he", "hemos", "hube", "hubiera", "hubierais", "hubieran",
    "hubieras", "hubieron", "hubiese", "hubieseis", "hubiesen", "hubieses", "hubimos",
    "hubiste", "hubisteis", "hubo", "la", "las", "le", "les", "lo", "los", "me",
    "mi", "mis", "mucho", "muchos", "muy", "mí", "mía", "mías", "mío", "míos",
    "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros",
    "o", "os", "otra", "otras", "otro", "otros", "para", "pero", "poco", "por",
    "porque", "que", "quien", "quienes", "qué", "se", "sea", "seamos", "sean",
    "seas", "seremos", "será", "serán", "serás", "seré", "seréis", "sería", "seríais",
    "seríamos", "serían", "serías", "seáis", "sido", "siendo", "sobre", "sois",
    "somos", "son", "soy", "su", "sus", "suya", "suyas", "suyo", "suyos", "también",
    "tan", "tanto", "te", "tendremos", "tendrá", "tendrán", "tendrás", "tendré",
    "tendréis", "tendría", "tendríais", "tendríamos", "tendrían", "tendrías", "tened",
    "tenemos", "tenga", "tengamos", "tengan", "tengas", "tengo", "tengáis", "tenida",
    "tenidas", "tenido", "tenidos", "teniendo", "tenéis", "tenía", "teníais",
    "teníamos", "tenían", "tenías", "ti", "tiene", "tienen", "tienes", "todo",
    "todos", "tu", "tus", "tuve", "tuviera", "tuvierais", "tuvieran", "tuvieras",
    "tuvieron", "tuviese", "tuvieseis", "tuviesen", "tuvieses", "tuvimos", "tuviste",
    "tuvisteis", "tuvo", "tuya", "tuyas", "tuyo", "tuyos", "tú", "un", "una", "uno",
    "unos", "vosotras", "vosotros", "vuestra", "vuestras", "vuestro", "vuestros",
    "y", "ya", "yo", "él", "éramos",
}

# Diccionario morfológico de lematización especializado en bitácoras de trading en español
SPANISH_TRADING_LEMMA_MAP: dict[str, str] = {
    # Verbos de entrada y ejecución
    "entré": "entrar", "entre": "entrar", "entrado": "entrar", "entrando": "entrar", "entrar": "entrar",
    "abrí": "abrir", "abri": "abrir", "abriendo": "abrir", "abierto": "abrir",
    "tomé": "tomar", "tome": "tomar", "tomando": "tomar",
    "ejecuté": "ejecutar", "ejecute": "ejecutar", "ejecutada": "ejecutar", "ejecutado": "ejecutar",
    "operé": "operar", "opere": "operar", "operando": "operar", "operar": "operar",
    "compré": "comprar", "compre": "comprar", "comprando": "comprar", "compras": "compra",
    "vendí": "vender", "vendi": "vender", "vendiendo": "vender", "ventas": "venta",
    # Verbos de paciencia / confirmación / espera
    "esperé": "esperar", "espere": "esperar", "esperando": "esperar", "esperado": "esperar", "esperar": "esperar",
    "aguardé": "aguardar", "aguarde": "aguardar", "aguardando": "aguardar",
    "confirmé": "confirmar", "confirme": "confirmar", "confirmarse": "confirmar", "confirmando": "confirmar",
    "respeté": "respetar", "respete": "respetar", "respetando": "respetar",
    "coloqué": "colocar", "coloque": "colocar", "colocando": "colocar", "poner": "colocar", "puesto": "colocar",
    "protegí": "proteger", "protegi": "proteger", "protegiendo": "proteger",
    "aseguré": "asegurar", "asegure": "asegurar", "asegurando": "asegurar",
    "gestioné": "gestionar", "gestione": "gestionar",
    # Verbos de error e indisciplina
    "apresuré": "apresurar", "apresure": "apresurar", "adelanté": "adelantar", "adelante": "adelantar",
    "adelantarme": "adelantar", "moví": "mover", "movi": "mover", "moviendo": "mover",
    "sobreoperé": "sobreoperar", "sobreopere": "sobreoperar", "sobreoperando": "sobreoperar",
    "cometí": "cometer", "cometi": "cometer", "cometiendo": "cometer",
    "caí": "caer", "cai": "caer", "cayó": "caer", "cayo": "caer", "cayendo": "caer",
    "subió": "subir", "subio": "subir", "subiendo": "subir", "sube": "subir",
    "rompió": "romper", "rompio": "romper", "rompiendo": "romper",
    "rebotó": "rebotar", "reboto": "rebotar", "rebotando": "rebotar",
    "tocó": "tocar", "toco": "tocar", "toque": "tocar",
    "marcó": "marcar", "marco": "marcar", "marcando": "marcar", "marcaba": "marcar",
    "llegó": "llegar", "llego": "llegar", "invalidó": "invalidar", "invalido": "invalidar",
    "siguió": "seguir", "siguio": "seguir", "desplomándose": "desplomar",
    "quemó": "quemar", "quemo": "quemar", "sacó": "sacar", "saco": "sacar",
    # Verbos de cierre y resultado
    "terminé": "terminar", "termine": "terminar", "terminando": "terminar",
    "cerré": "cerrar", "cerre": "cerrar", "cerró": "cerrar", "cerro": "cerrar",
    "cerrara": "cerrar", "cerrando": "cerrar",
    "salí": "salir", "sali": "salir", "saliendo": "salir",
    "alcancé": "alcanzar", "alcance": "alcanzar",
    "perdiendo": "perder", "perdí": "perder", "perdi": "perder", "perdido": "perder",
    "pérdida": "perdida", "pérdidas": "perdida", "perdidas": "perdida",
    "ganando": "ganar", "gané": "ganar", "gane": "ganar", "ganado": "ganar",
    "ganadora": "ganador", "ganadoras": "ganador", "ganancias": "ganancia",
    "perdedora": "perdedor", "perdedoras": "perdedor",
    # Sustantivos y adjetivos técnicos de trading
    "velas": "vela", "cierres": "cierre", "bandas": "banda", "medias": "media",
    "móvil": "movil", "móviles": "movil", "alcistas": "alcista", "bajistas": "bajista",
    "confirmaciones": "confirmacion", "confirmación": "confirmacion",
    "operaciones": "operacion", "operación": "operacion",
    "reglas": "regla", "errores": "error", "índices": "indice", "índice": "indice",
    "sintéticos": "sintetico", "sintético": "sintetico",
}

WIN_EMOJIS = {"✅", "🟢", "🚀", "📈", "🐂", "💰", "🎯"}
LOSS_EMOJIS = {"❌", "🔴", "📉", "🩸", "🐻", "💀", "⚠️"}


def strip_accents(text: str) -> str:
    """Elimina tildes/diacríticos manteniendo la letra ñ."""
    protected = text.replace("ñ", "__enie__").replace("Ñ", "__ENIE__")
    normalized = unicodedata.normalize("NFD", protected)
    without_diacritics = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_diacritics.replace("__enie__", "ñ").replace("__ENIE__", "Ñ")


def _fallback_spanish_stem(word: str) -> str:
    """Stemmer determinista de respaldo en español si NLTK no está instalado."""
    if word.startswith("_") and word.endswith("_"):
        return word
    w = strip_accents(word.lower())
    if len(w) <= 3 or w.isdigit():
        return w
    suffixes = (
        "amientos", "imiento", "aciones", "adoras", "adores", "logias",
        "mente", "iendo", "ando", "aban", "aran", "ieron", "aste", "iste",
        "acion", "antes", "ancia", "adora", "ador", "ales", "able", "ible",
        "anza", "idad", "ivas", "ivos", "osa", "oso", "aba", "ada", "ado",
        "ara", "are", "ase", "ida", "ido", "ar", "er", "ir", "as", "es", "os", "a", "e", "o",
    )
    for suf in suffixes:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


class TradingTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Pipeline de preprocesamiento de NLP y extractor estructurado para bitácoras de trading.
    Se integra directamente dentro de `sklearn.pipeline.Pipeline` para garantizar que el
    mismo preprocesamiento del entrenamiento se ejecute idéntico en el Backend (Sección 5.4 PDF).
    """

    def __init__(self, mode: str = "lemma", keep_negations: bool = True) -> None:
        self.mode = mode
        self.keep_negations = keep_negations

    def get_stopwords(self) -> set[str]:
        """Devuelve el conjunto efectivo de stop words respetando las negaciones protegidas."""
        stopwords = set(SPANISH_BASE_STOPWORDS) | set(CUSTOM_EXTRA_STOPWORDS)
        if self.keep_negations:
            stopwords = stopwords - NEGATION_AND_TRADING_KEEP
            stopwords = stopwords - {strip_accents(w) for w in NEGATION_AND_TRADING_KEEP}
        return stopwords

    # ------------------------------------------------------------------
    # PASO 1: NORMALIZACIÓN (Sección 5.2 PDF)
    # ------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        """
        Normaliza el texto de trading:
        - Convierte a minúsculas y elimina URLs/HTML.
        - Traduce emojis operativos a tokens semánticos (`_emoji_ganancia_`, `_emoji_perdida_`).
        - Detecta niveles numéricos del RSI (<30 sobreventa, >70 sobrecompra) e inyecta tokens semánticos.
        - Elimina signos de puntuación generales conservando guiones bajos y números relevantes.
        """
        if not isinstance(text, str):
            text = str(text or "")

        cleaned = text.strip().lower()

        # 1. Eliminar URLs y etiquetas HTML
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        # 2. Traducir emojis operativos de trading a tokens semánticos
        for em in WIN_EMOJIS:
            if em in cleaned:
                cleaned = cleaned.replace(em, " _emoji_ganancia_ ")
        for em in LOSS_EMOJIS:
            if em in cleaned:
                cleaned = cleaned.replace(em, " _emoji_perdida_ ")

        # 3. Unificar nombres compuestos de índices sintéticos y expresiones clave
        cleaned = re.sub(r"\bvolatility\s*100\b|\bv\s*100\b", "volatility_100", cleaned)
        cleaned = re.sub(r"\bvolatility\s*75\b|\bv\s*75\b", "volatility_75", cleaned)
        cleaned = re.sub(r"\bboom\s*1000\b", "boom_1000", cleaned)
        cleaned = re.sub(r"\bcrash\s*500\b", "crash_500", cleaned)
        cleaned = re.sub(r"\bstep\s*index\b", "step_index", cleaned)
        cleaned = re.sub(r"\bbreak\s*[- ]?\s*even\b", "break_even", cleaned)

        # 4. Detectar nivel numérico del RSI e inyectar token semántico de zona técnica
        def _inject_rsi_zone(match: re.Match[str]) -> str:
            prefix = match.group(1)
            val_str = match.group(2)
            try:
                val = float(val_str)
                if val <= 30:
                    return f"{prefix}{val_str} _nivel_sobreventa_"
                if val >= 70:
                    return f"{prefix}{val_str} _nivel_sobrecompra_"
                return f"{prefix}{val_str} _nivel_rsi_medio_"
            except ValueError:
                return match.group(0)

        cleaned = re.sub(
            r"(\brsi\b(?:\s+\w+){0,3}?\s+)(\d{1,2}(?:\.\d+)?)\b",
            _inject_rsi_zone,
            cleaned,
        )

        # 5. Limpiar signos de puntuación y caracteres no alfanuméricos (conservando tildes, ñ y _)
        cleaned = re.sub(r"[^\w\sáéíóúüñ_]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    # ------------------------------------------------------------------
    # PASO 2: TOKENIZACIÓN (Sección 5.2 PDF)
    # ------------------------------------------------------------------
    def tokenize(self, normalized_text: str) -> list[str]:
        """Separa el texto normalizado en tokens preservando siglas técnicas y tokens semánticos."""
        if not normalized_text:
            return []
        return re.findall(r"[a-záéíóúüñ0-9_]+", normalized_text.lower())

    # ------------------------------------------------------------------
    # PASO 3: STOP WORDS (Sección 5.2 PDF)
    # ------------------------------------------------------------------
    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Elimina stop words del español conservando las negaciones críticas ('no', 'sin', etc.)."""
        stopwords = self.get_stopwords()
        filtered: list[str] = []
        for tok in tokens:
            tok_no_accent = strip_accents(tok)
            if tok in stopwords or tok_no_accent in stopwords:
                continue
            filtered.append(tok)
        return filtered

    # ------------------------------------------------------------------
    # PASO 4: STEMMING Y LEMATIZACIÓN (Sección 5.2 PDF)
    # ------------------------------------------------------------------
    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """Aplica Stemming en español (SnowballStemmer)."""
        stemmed: list[str] = []
        for tok in tokens:
            if tok.startswith("_") and tok.endswith("_"):
                stemmed.append(tok)
            elif _NLTK_STEMMER is not None:
                stemmed.append(_NLTK_STEMMER.stem(tok))
            else:
                stemmed.append(_fallback_spanish_stem(tok))
        return stemmed

    def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """
        Aplica Lematización en español para dominio de trading y normaliza las tildes
        sobre el lema final para unificar escritura con y sin acento.
        """
        lemmatized: list[str] = []
        for tok in tokens:
            if tok.startswith("_") and tok.endswith("_"):
                lemmatized.append(tok)
                continue
            if tok in SPANISH_TRADING_LEMMA_MAP:
                lemma = SPANISH_TRADING_LEMMA_MAP[tok]
                lemmatized.append(strip_accents(lemma))
                continue
            tok_plain = strip_accents(tok)
            if tok_plain in SPANISH_TRADING_LEMMA_MAP:
                lemma = SPANISH_TRADING_LEMMA_MAP[tok_plain]
                lemmatized.append(strip_accents(lemma))
                continue

            # Reglas morfológicas para gerundios, participios y plurales regulares
            if len(tok_plain) > 5 and tok_plain.endswith("ando"):
                lemmatized.append(tok_plain[:-4] + "ar")
            elif len(tok_plain) > 5 and tok_plain.endswith("iendo"):
                lemmatized.append(tok_plain[:-5] + "er")
            elif len(tok_plain) > 4 and tok_plain.endswith("es") and not tok_plain.endswith("mes"):
                lemmatized.append(tok_plain[:-2])
            elif len(tok_plain) > 4 and tok_plain.endswith("s") and not tok_plain.endswith(("rs", "is", "us")):
                lemmatized.append(tok_plain[:-1])
            else:
                lemmatized.append(tok_plain)
        return lemmatized

    def stem_or_lemmatize(self, tokens: list[str]) -> list[str]:
        """Ejecuta Lematización o Stemming según el parámetro `self.mode` ('lemma' o 'stem')."""
        if self.mode == "stem":
            return self.stem_tokens(tokens)
        return self.lemmatize_tokens(tokens)

    def process_text(self, text: str) -> str:
        """Ejecuta el pipeline completo sobre un texto y devuelve el string listo para TF-IDF."""
        norm = self.normalize(text)
        tokens = self.tokenize(norm)
        no_stop = self.remove_stopwords(tokens)
        final_tokens = self.stem_or_lemmatize(no_stop)
        return " ".join(final_tokens)

    # ------------------------------------------------------------------
    # INTERFAZ SCIKIT-LEARN PIPELINE (fit / transform)
    # ------------------------------------------------------------------
    def fit(self, X: Iterable[str], y: Any = None) -> "TradingTextPreprocessor":
        return self

    def transform(self, X: Iterable[str]) -> list[str]:
        return [self.process_text(text) for text in X]

    # ------------------------------------------------------------------
    # DESGLOSE EXPLICATIVO PASO A PASO PARA EL BACKEND Y FRONTEND
    # ------------------------------------------------------------------
    def explain_steps(self, text: str) -> dict[str, Any]:
        """
        Devuelve el Antes y Después exacto de cada paso del pipeline de NLP
        para ser mostrado en el endpoint `/api/v1/predict` y en la UI del Frontend.
        """
        norm = self.normalize(text)
        tokens = self.tokenize(norm)
        stopwords = self.get_stopwords()

        removed_stopwords: list[str] = []
        stopwords_removed: list[str] = []
        kept_negations: list[str] = []

        for tok in tokens:
            tok_plain = strip_accents(tok)
            if tok in stopwords or tok_plain in stopwords:
                removed_stopwords.append(tok)
            else:
                stopwords_removed.append(tok)
                if tok in NEGATION_AND_TRADING_KEEP or tok_plain in NEGATION_AND_TRADING_KEEP:
                    kept_negations.append(tok)

        stemmed_preview = self.stem_tokens(stopwords_removed)
        lemmatized_preview = self.lemmatize_tokens(stopwords_removed)
        final_tokens = stemmed_preview if self.mode == "stem" else lemmatized_preview

        return {
            "raw_text": text,
            "normalized": norm,
            "tokens": tokens,
            "stopwords_removed": stopwords_removed,
            "removed_stopwords_list": removed_stopwords,
            "kept_negations": kept_negations,
            "lemmatized_or_stemmed": final_tokens,
            "stemmed_preview": stemmed_preview,
            "lemmatized_preview": lemmatized_preview,
            "active_mode": self.mode,
        }

    # ------------------------------------------------------------------
    # EXTRACCIÓN DE INFORMACIÓN ESTRUCTURADA DE TRADING (Caso Ing. Kenji)
    # ------------------------------------------------------------------
    def extract_trading_entities(self, text: str) -> dict[str, Any]:
        """
        Identifica y estructura a partir de la bitácora del trader:
        - Indicador técnico (RSI + valor numérico, MACD, EMA, Bollinger, Soporte/Resistencia)
        - Acción realizada (Compra / Venta)
        - Mercado / Activo sintético (Volatility 100, Volatility 75, Boom 1000, Crash 500, etc.)
        - Temporalidad (M1, M5, M15, H1)
        - Error operativo cometido (No esperar cierre de vela, Sin Stop Loss, FOMO, etc.)
        - Resultado operativo detectado en el texto
        """
        lower = strip_accents(text.lower())

        # 1. Detectar Indicador Técnico y nivel numérico
        indicator_label = "Acción del Precio (Price Action)"
        indicator_code = "Price_Action"
        rsi_value: int | None = None

        rsi_match = re.search(r"\brsi\b(?:\s+\w+){0,3}?\s+(\d{1,2})\b", lower)
        if "rsi" in lower:
            indicator_code = "RSI"
            if rsi_match:
                rsi_value = int(rsi_match.group(1))
                if rsi_value <= 30:
                    zone = "Sobreventa"
                elif rsi_value >= 70:
                    zone = "Sobrecompra"
                else:
                    zone = "Zona Media"
                indicator_label = f"RSI (Nivel {rsi_value} - {zone})"
            else:
                indicator_label = "RSI (Relative Strength Index)"
        elif "macd" in lower:
            indicator_code = "MACD"
            indicator_label = "MACD (Cruce / Histograma)"
        elif "ema" in lower or "media movil" in lower:
            indicator_code = "EMA"
            ema_nums = re.findall(r"\bema\s*(\d{2,3})\b", lower)
            if ema_nums:
                indicator_label = f"EMA ({', '.join(ema_nums)})"
            else:
                indicator_label = "EMA (Media Móvil Exponencial)"
        elif "bollinger" in lower or "banda" in lower:
            indicator_code = "Bollinger"
            indicator_label = "Bandas de Bollinger"
        elif "soporte" in lower or "resistencia" in lower or "fibonacci" in lower:
            indicator_code = "Soporte_Resistencia"
            indicator_label = "Soporte / Resistencia & Fibonacci"

        # 2. Detectar Acción realizada (Compra vs Venta)
        action = "Seguimiento / Neutral"
        if any(w in lower for w in ("compra", "compre", "largo", "buy", "long")):
            action = "Compra"
        elif any(w in lower for w in ("venta", "vendi", "corto", "sell", "short")):
            action = "Venta"

        # 3. Detectar Activo / Mercado Sintético
        market = "Volatility 100"
        if "volatility 75" in lower or "v75" in lower:
            market = "Volatility 75"
        elif "boom 1000" in lower or "boom" in lower:
            market = "Boom 1000"
        elif "crash 500" in lower or "crash" in lower:
            market = "Crash 500"
        elif "step index" in lower:
            market = "Step Index"
        elif "btc" in lower or "bitcoin" in lower:
            market = "BTC/USD"
        elif "volatility 100" in lower or "v100" in lower:
            market = "Volatility 100"

        # 4. Detectar Temporalidad
        tf_match = re.search(r"\b(m1|m5|m15|m30|h1|h4)\b", lower)
        timeframe = tf_match.group(1).upper() if tf_match else "M5"

        # 5. Detectar Error Operativo
        operational_error = "Ninguno (Disciplinado)"
        if (
            ("no espere" in lower or "no esperar" in lower or "sin esperar" in lower or "antes de tiempo" in lower)
            and ("vela" in lower or "cierre" in lower or "confirmacion" in lower)
        ):
            operational_error = "No esperar cierre de vela"
        elif "sin stop loss" in lower or "sin colocar stop" in lower or "sin poner stop" in lower or "sin sl" in lower:
            operational_error = "Operar sin Stop Loss"
        elif "movi el stop" in lower or "mover el stop" in lower or "movi mi stop" in lower:
            operational_error = "Mover Stop Loss"
        elif "fomo" in lower or "impulso" in lower or "ansiedad" in lower or "compre en el techo" in lower:
            operational_error = "FOMO / Entrada anticipada"
        elif "sobreopere" in lower or "sobreoperar" in lower or "revancha" in lower or "sexta compra" in lower:
            operational_error = "Sobreoperar"
        elif "contra la tendencia" in lower or "contra tendencia" in lower:
            operational_error = "Contra tendencia"

        # 6. Detectar Resultado explícito en la bitácora
        outcome_detected = "Break-Even / Neutro"
        if any(w in lower for w in ("perdiendo", "perdida", "perdi", "quemo", "saco el stop")):
            outcome_detected = "Pérdida (Loss)"
        elif any(w in lower for w in ("ganancia", "ganando", "ganadora", "profit", "gane")):
            outcome_detected = "Ganancia (Win)"

        return {
            "indicator": indicator_label,
            "indicator_code": indicator_code,
            "rsi_value": rsi_value,
            "action": action,
            "market": market,
            "timeframe": timeframe,
            "operational_error": operational_error,
            "outcome_detected": outcome_detected,
        }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sample = "Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo ❌"
    preprocessor = TradingTextPreprocessor(mode="lemma")
    steps = preprocessor.explain_steps(sample)
    entities = preprocessor.extract_trading_entities(sample)

    print("=== PRUEBA DE TradingTextPreprocessor ===")
    for k, v in steps.items():
        print(f"  {k:<24}: {v}")
    print("\n=== ENTIDADES EXTRAÍDAS DE LA BITÁCORA ===")
    for k, v in entities.items():
        print(f"  {k:<24}: {v}")
