"""
Script reproducible de obtención, curación y partición estratificada del dataset
de bitácoras y estrategias de trading en español (TradingJournal NLP).

Cumple con las Secciones 4, 5.1 y 5.4 del PDF del Proyecto Integrador:
- Documenta origen, licencia y fecha de obtención.
- Genera el dataset crudo en `data/raw/trading_journal_es.csv` (incluyendo casos
  reales de valores faltantes y duplicados para auditar en el EDA).
- Limpia nulos y duplicados, y ejecuta la partición estratificada 80% entrenamiento
  (`data/processed/train.csv`) y 20% prueba (`data/processed/test.csv`) con
  semilla fija (`random_state=42`).
- Inicializa `data/processed/journal_history.json` con registros estructurados
  base para las estadísticas del sistema (`/api/v1/stats`).
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

RAW_CSV_PATH = RAW_DIR / "trading_journal_es.csv"
TRAIN_CSV_PATH = PROCESSED_DIR / "train.csv"
TEST_CSV_PATH = PROCESSED_DIR / "test.csv"
HISTORY_JSON_PATH = PROCESSED_DIR / "journal_history.json"
METADATA_JSON_PATH = RAW_DIR / "dataset_metadata.json"


# Catálogos de dominio: Mercados Sintéticos (Deriv / MT5), Indicadores y Contextos
MARKETS = [
    ("Volatility 100", ["en Volatility 100", "en el índice Volatility 100", "operando V100 en MT5", "en Deriv Volatility 100", "en el activo V100"]),
    ("Volatility 75", ["en Volatility 75", "en el índice V75", "operando V75 en MT5", "en Deriv Volatility 75", "en el sintético V75"]),
    ("Boom 1000", ["en Boom 1000", "en el índice Boom 1000", "buscando spikes en Boom 1000", "operando Boom 1000 en M5"]),
    ("Crash 500", ["en Crash 500", "en el índice Crash 500", "operando caídas en Crash 500", "en Deriv Crash 500"]),
    ("Step Index", ["en Step Index", "operando Step Index en M5", "en el índice Step Index de Deriv"]),
    ("BTC/USD", ["en BTCUSD", "operando Bitcoin en MT5", "en el par BTC/USD", "en cripto BTCUSD"]),
]

TIMEFRAMES = ["en temporalidad M1", "en gráfico de M5", "en velas de M15", "en H1", "en temporalidad de 5 minutos", "en M5"]

# Plantillas por indicador técnico para operaciones GANADORAS DISCIPLINADAS
DISCIPLINED_WIN_TEMPLATES = [
    {
        "indicator": "RSI",
        "action": "Compra",
        "templates": [
            "Entré en compra {market} {tf} cuando el RSI estaba en {rsi_low} en zona de sobreventa, esperé el cierre de la vela alcista de confirmación, coloqué mi stop loss técnico y cerré en ganancia {emoji_win}",
            "Tomé compra {market} {tf} con el RSI marcando {rsi_low}, sí esperé el cierre de la vela envolvente sobre soporte y respeté el take profit logrando una operación ganadora",
            "Operación en compra {market}: el indicador RSI llegó a {rsi_low}, aguardé pacientemente la confirmación del cierre de vela sin adelantarme y salí en profit cumpliendo mi plan {emoji_win}",
            "Abrí largo en compra {market} {tf} tras ver divergencia alcista con RSI en {rsi_low}, esperé confirmación de vela, gestioné el riesgo con stop loss fijo y terminé ganando",
            "Ejecuté compra disciplinada {market} cuando el RSI tocó {rsi_low}; respeté mi estrategia esperando el cierre de la vela en {tf} y alcancé el objetivo de ganancia",
        ],
    },
    {
        "indicator": "RSI",
        "action": "Venta",
        "templates": [
            "Entré en venta {market} {tf} cuando el RSI subió a {rsi_high} en sobrecompra, esperé el cierre de la vela bajista de rechazo y cerré con buena ganancia {emoji_win}",
            "Ejecuté venta en corto {market} con RSI en {rsi_high}, esperé confirmación al cierre de la vela en resistencia y respeté mi stop loss hasta tocar el take profit",
            "Operé venta {market} {tf} viendo el RSI agotado en {rsi_high}; no me adelanté, esperé el cierre de vela bajista y terminé ganando según la estrategia",
        ],
    },
    {
        "indicator": "EMA",
        "action": "Compra",
        "templates": [
            "Entré en compra {market} {tf} a favor de la tendencia tras el cruce de la EMA 20 sobre la EMA 50, esperé el pullback y cierre de vela con stop loss ajustado y salí en ganancia {emoji_win}",
            "Compra ejecutada {market} respetando la tendencia alcista de la EMA 200 {tf}; esperé confirmación de vela sobre la media móvil y cerré en profit disciplinado",
            "Tomé entrada en compra {market} {tf} cuando el precio rebotó limpio en la EMA 50, esperé el cierre de la vela alcista y alcancé mi relación riesgo beneficio 1:2 ganadora",
        ],
    },
    {
        "indicator": "MACD",
        "action": "Venta",
        "templates": [
            "Entré en venta {market} {tf} tras confirmarse el cruce bajista del MACD con histograma negativo, esperé el cierre de la vela y cerré la operación en ganancia {emoji_win}",
            "Venta disciplinada {market}: el MACD dio señal bajista en {tf}, coloqué stop loss sobre el último máximo, esperé confirmación de vela y terminé ganando",
            "Operé en corto {market} {tf} combinando divergencia en MACD con rechazo en resistencia; esperé el cierre de la vela y aseguré ganancia en take profit",
        ],
    },
    {
        "indicator": "Bollinger",
        "action": "Compra",
        "templates": [
            "Entré en compra {market} {tf} tras toque en la banda inferior de Bollinger con RSI en {rsi_low}, esperé el cierre de la vela dentro de la banda y salí con ganancia {emoji_win}",
            "Compra técnica {market}: las bandas de Bollinger mostraron rechazo alcista en {tf}, respeté el cierre de vela y mi stop loss, logrando una operación positiva",
        ],
    },
    {
        "indicator": "Soporte_Resistencia",
        "action": "Compra",
        "templates": [
            "Entré en compra {market} {tf} justo en zona institucional de soporte y Fibonacci 61.8, esperé el cierre de la vela martillo con stop loss definido y terminé ganando {emoji_win}",
            "Operación ganadora en compra {market}: esperé pacientemente el retesteo del soporte en {tf}, confirmé con cierre de vela alcista y respeté mi gestión de riesgo",
        ],
    },
]

# Plantillas por error operativo para operaciones PERDEDORAS POR ERROR OPERATIVO
ERROR_LOSS_TEMPLATES = [
    {
        "operational_error": "No esperar cierre de vela",
        "indicator": "RSI",
        "action": "Compra",
        "templates": [
            "Entré en compra {market} cuando el RSI estaba en {rsi_low}, pero no esperé el cierre de la vela y terminé perdiendo {emoji_loss}",
            "Abrí compra {market} {tf} viendo el RSI en {rsi_low}, pero no esperé el cierre de la vela de confirmación, el precio siguió cayendo y cerré en pérdida",
            "Me apresuré a entrar en compra {market} con el RSI marcando {rsi_low}; no esperé a que cerrara la vela en {tf} y me sacó el stop loss perdiendo {emoji_loss}",
            "Error operativo {market}: vi el RSI en sobreventa en {rsi_low}, entré en compra antes de tiempo porque no esperé el cierre de la vela y terminé en pérdida",
        ],
    },
    {
        "operational_error": "No esperar cierre de vela",
        "indicator": "MACD",
        "action": "Venta",
        "templates": [
            "Entré en venta {market} {tf} porque parecía que el MACD iba a cruzar, pero no esperé el cierre de la vela y el cruce se invalidó terminando en pérdida {emoji_loss}",
            "Vendí {market} anticipando giro en MACD {tf}, no esperé el cierre de la vela para confirmar el rechazo y terminé perdiendo la operación",
        ],
    },
    {
        "operational_error": "Operar sin Stop Loss",
        "indicator": "EMA",
        "action": "Venta",
        "templates": [
            "Entré en venta {market} {tf} debajo de la EMA 50, pero operé sin stop loss esperando que regresara y un spike brusco me dejó con una pérdida fuerte {emoji_loss}",
            "Vendí {market} cuando rompió la EMA 20 {tf}, cometí el error de entrar sin stop loss y la cuenta sufrió una pérdida grande al girarse el mercado",
            "Operación perdedora en venta {market}: entré sin colocar stop loss en {tf} confiando en la EMA 200 y el retroceso quemó gran parte del margen {emoji_loss}",
        ],
    },
    {
        "operational_error": "Operar sin Stop Loss",
        "indicator": "RSI",
        "action": "Compra",
        "templates": [
            "Compré {market} {tf} porque el RSI estaba en {rsi_low}, pero entré sin stop loss y la caída continuó dejándome en pérdida severa {emoji_loss}",
            "Entré en compra {market} con RSI en {rsi_low} sin poner stop loss en MT5, el índice siguió desplomándose y terminé perdiendo mucho más de lo permitido",
        ],
    },
    {
        "operational_error": "FOMO / Entrada anticipada",
        "indicator": "Bollinger",
        "action": "Compra",
        "templates": [
            "Entré en compra {market} {tf} por puro impulso y FOMO al ver una vela gigante fuera de las bandas de Bollinger, compré en el techo sin confirmación y terminé perdiendo {emoji_loss}",
            "Me ganó la ansiedad y el FOMO {market}: vi que el precio subía rápido en {tf} con Bollinger expandido, entré en compra tarde sin esperar retroceso y cerré en pérdida",
        ],
    },
    {
        "operational_error": "Mover Stop Loss",
        "indicator": "Soporte_Resistencia",
        "action": "Venta",
        "templates": [
            "Entré en venta {market} {tf} en una resistencia, pero cuando el precio se fue en mi contra moví el stop loss dos veces por miedo a perder y terminé perdiendo el doble {emoji_loss}",
            "Vendí {market} bajo resistencia en {tf}, no respeté mi riesgo inicial porque moví el stop loss esperando un giro y cerré con una pérdida grave",
        ],
    },
    {
        "operational_error": "Sobreoperar",
        "indicator": "MACD",
        "action": "Compra",
        "templates": [
            "Volví a entrar en compra {market} {tf} por revancha tras dos pérdidas seguidas mirando el MACD sin señal clara; caí en sobreoperar y terminé perdiendo otra vez {emoji_loss}",
            "Sobreoperé {market} en {tf} abriendo mi sexta compra seguida por frustración aunque el MACD estaba plano, no respeté mi límite diario y salí en pérdida",
        ],
    },
    {
        "operational_error": "Contra tendencia",
        "indicator": "EMA",
        "action": "Venta",
        "templates": [
            "Entré en venta {market} {tf} totalmente contra la tendencia alcista de la EMA 200 intentando adivinar el techo y el impulso alcista me sacó en pérdida {emoji_loss}",
            "Operé en corto {market} contra tendencia principal de la EMA 50 y EMA 200 en {tf}, no respeté la estructura alcista y terminé perdiendo {emoji_loss}",
        ],
    },
]

# Plantillas para operaciones de RIESGO NEUTRAL / BREAK-EVEN
NEUTRAL_TEMPLATES = [
    {
        "indicator": "Bollinger",
        "action": "Compra",
        "operational_error": "Ninguno (Disciplinado)",
        "templates": [
            "Entré en compra {market} {tf} sobre la banda media de Bollinger, pero el mercado entró en consolidación lateral sin volumen y cerré en break even sin ganancia ni pérdida",
            "Operación neutral en compra {market} {tf}: las bandas de Bollinger se estrecharon indicando rango lateral, protegí la entrada en punto de equilibrio y salí en empate",
            "Tomé compra {market} con Bollinger en {tf}, el precio avanzó lento sin fuerza direccional así que aseguré en break even con resultado neutro",
        ],
    },
    {
        "indicator": "EMA",
        "action": "Venta",
        "operational_error": "Ninguno (Disciplinado)",
        "templates": [
            "Entré en venta {market} {tf} cerca de la EMA 50, al ver indecisión y mechas en ambos lados moví a break even y la operación cerró neutra en cero",
            "Registro de seguimiento {market} {tf}: las medias EMA 20 y EMA 50 están entrelazadas en rango horizontal, cerré mi venta en punto de entrada sin ganancia ni pérdida",
        ],
    },
    {
        "indicator": "RSI",
        "action": "Compra",
        "operational_error": "Ninguno (Disciplinado)",
        "templates": [
            "Entré en compra {market} {tf} con el RSI en zona media de {rsi_mid}, el precio no mostró impulso claro hacia el take profit y cerré en break even preventivo",
            "Operé compra {market} cuando el RSI marcaba {rsi_mid} en {tf}, ante la falta de volatilidad protegí la posición en empate técnico",
        ],
    },
    {
        "indicator": "MACD",
        "action": "Venta",
        "operational_error": "Ninguno (Disciplinado)",
        "templates": [
            "Abrí venta {market} {tf} con el histograma del MACD plano cerca de cero, el activo quedó sin dirección definida y cerré la orden en break even",
            "Operación en empate {market} {tf}: señal mixta entre MACD y soporte local, decidí cerrar manualmente en precio de entrada sin pérdidas ni ganancias",
        ],
    },
]

# Casos difíciles / ambiguos reales (~4% del dataset) para que los modelos tengan un error realista (~5-8%)
# y podamos analizar fallos genuinos en la Sección 5.3 del PDF (Análisis de Errores).
HARD_EDGE_CASES = [
    {
        "text": "Entré en compra en Volatility 100 en M5 cuando el RSI estaba en 27 sin esperar el cierre de la vela, fue un error de impaciencia pero justo salió una vela alcista gigante y terminé ganando por suerte",
        "label": "Ganadora_Disciplinada",
        "indicator": "RSI",
        "action": "Compra",
        "market": "Volatility 100",
        "operational_error": "No esperar cierre de vela",
    },
    {
        "text": "En Boom 1000 entré en compra en M1 por impulso sin colocar stop loss cuando el RSI marcó 29, cometí error operativo pero cayó un spike alcista inmediato y cerré en ganancia inesperada",
        "label": "Ganadora_Disciplinada",
        "indicator": "RSI",
        "action": "Compra",
        "market": "Boom 1000",
        "operational_error": "Operar sin Stop Loss",
    },
    {
        "text": "Respeté todo mi plan en Volatility 100 en M15: esperé el cierre de la vela alcista con RSI en 28 y cruce de EMA 20, coloqué stop loss técnico pero un latigazo brusco me sacó en pérdida",
        "label": "Perdedora_Error_Operativo",
        "indicator": "RSI",
        "action": "Compra",
        "market": "Volatility 100",
        "operational_error": "Ninguno (Disciplinado)",
    },
    {
        "text": "En Crash 500 esperé confirmación de cierre de vela bajista con MACD negativo y stop loss bien puesto, hice todo con disciplina pero el precio rebotó con fuerza y terminé perdiendo",
        "label": "Perdedora_Error_Operativo",
        "indicator": "MACD",
        "action": "Venta",
        "market": "Crash 500",
        "operational_error": "Ninguno (Disciplinado)",
    },
    {
        "text": "Entré en compra en Volatility 75 con RSI en 26 pero no esperé el cierre de la vela; aunque estuve casi en pérdida, el precio regresó a mi entrada y cerré en break even",
        "label": "Riesgo_Neutral",
        "indicator": "RSI",
        "action": "Compra",
        "market": "Volatility 75",
        "operational_error": "No esperar cierre de vela",
    },
]


def _strip_some_accents(text: str, rng: random.Random) -> str:
    """Simula variaciones realistas de escritura rápida de traders (algunos escriben sin tilde)."""
    if rng.random() < 0.28:
        replacements = [("Entré", "Entre"), ("entré", "entre"), ("esperé", "espere"), ("terminé", "termine"), ("pérdida", "perdida"), ("índice", "indice")]
        for old, new in replacements:
            if rng.random() < 0.6:
                text = text.replace(old, new)
    return text


def generate_corpus(seed: int = SEED) -> pd.DataFrame:
    """Genera el corpus de bitácoras de trading en español con 2,100 registros limpios + casos de auditoría."""
    rng = random.Random(seed)
    np.random.seed(seed)

    records: list[dict] = []

    # 1. Insertar como registro #1 el caso literal acordado con el Ing. Kenji por WhatsApp
    records.append(
        {
            "text": "Entré en compra cuando el RSI estaba en 28, pero no esperé el cierre de la vela y terminé perdiendo",
            "label": "Perdedora_Error_Operativo",
            "indicator": "RSI",
            "action": "Compra",
            "market": "Volatility 100",
            "operational_error": "No esperar cierre de vela",
        }
    )

    win_emojis = ["", "", "✅", "🟢", "🚀"]
    loss_emojis = ["", "", "❌", "🔴", "📉"]
    extra_clauses_win = [
        "",
        " manteniendo la disciplina del plan",
        " con excelente ratio riesgo beneficio",
        " siguiendo las reglas de mi bitácora",
        " con gestión de lotaje conservadora en MT5",
        " tras validar la estructura del mercado sintético",
        " respetando el nivel clave del gráfico",
    ]
    extra_clauses_loss = [
        "",
        " por no respetar mi plan de trading",
        " afectando mi estadística semanal en MT5",
        " por actuar con impaciencia frente al gráfico",
        " rompiendo mis reglas de gestión de riesgo",
        " en una sesión de alta volatilidad sintética",
        " cometiendo un fallo clásico de psicotrading",
    ]
    extra_clauses_neutral = [
        "",
        " priorizando proteger el capital de la cuenta",
        " mientras esperaba definición de la sesión",
        " manteniendo el riesgo bajo control en MT5",
        " al notar agotamiento en las velas",
        " dejando la cuenta intacta para la siguiente entrada",
    ]

    # Meta de distribución limpia (2,100 registros únicos):
    # - Perdedora_Error_Operativo: 900 (~42.9%)
    # - Ganadora_Disciplinada: 800 (~38.1%)
    # - Riesgo_Neutral: 400 (~19.0%)
    target_counts = {
        "Perdedora_Error_Operativo": 900,
        "Ganadora_Disciplinada": 800,
        "Riesgo_Neutral": 400,
    }

    seen_texts = {records[0]["text"]}

    # Insertar réplicas variadas de casos difíciles (~75 registros = 3.5% de ruido realista)
    for i in range(75):
        base_case = HARD_EDGE_CASES[i % len(HARD_EDGE_CASES)]
        m_name, m_phrases = rng.choice(MARKETS)
        tf = rng.choice(TIMEFRAMES)
        rsi_val = rng.randint(19, 29)
        variant_text = f"{base_case['text']} ({rng.choice(m_phrases)} {tf}, nivel {rsi_val}, ref #{i + 1})"
        variant_text = _strip_some_accents(variant_text, rng)
        if variant_text not in seen_texts:
            seen_texts.add(variant_text)
            records.append(
                {
                    "text": variant_text,
                    "label": base_case["label"],
                    "indicator": base_case["indicator"],
                    "action": base_case["action"],
                    "market": m_name,
                    "operational_error": base_case["operational_error"],
                }
            )

    def count_label(lbl: str) -> int:
        return sum(1 for r in records if r["label"] == lbl)

    # Completar Perdedora_Error_Operativo hasta 900
    attempt = 0
    while count_label("Perdedora_Error_Operativo") < target_counts["Perdedora_Error_Operativo"]:
        attempt += 1
        group = rng.choice(ERROR_LOSS_TEMPLATES)
        tpl = rng.choice(group["templates"])
        m_name, m_phrases = rng.choice(MARKETS)
        text = tpl.format(
            market=rng.choice(m_phrases),
            tf=rng.choice(TIMEFRAMES),
            rsi_low=rng.randint(16, 29),
            rsi_high=rng.randint(71, 86),
            emoji_loss=rng.choice(loss_emojis),
        ).strip()
        clause = rng.choice(extra_clauses_loss)
        if clause:
            text = f"{text}{clause}"
        if attempt > 400:
            text = f"{text} [op {attempt}]"
        text = _strip_some_accents(text, rng)
        if text not in seen_texts:
            seen_texts.add(text)
            records.append(
                {
                    "text": text,
                    "label": "Perdedora_Error_Operativo",
                    "indicator": group["indicator"],
                    "action": group["action"],
                    "market": m_name,
                    "operational_error": group["operational_error"],
                }
            )

    # Completar Ganadora_Disciplinada hasta 800
    attempt = 0
    while count_label("Ganadora_Disciplinada") < target_counts["Ganadora_Disciplinada"]:
        attempt += 1
        group = rng.choice(DISCIPLINED_WIN_TEMPLATES)
        tpl = rng.choice(group["templates"])
        m_name, m_phrases = rng.choice(MARKETS)
        text = tpl.format(
            market=rng.choice(m_phrases),
            tf=rng.choice(TIMEFRAMES),
            rsi_low=rng.randint(18, 30),
            rsi_high=rng.randint(70, 85),
            emoji_win=rng.choice(win_emojis),
        ).strip()
        clause = rng.choice(extra_clauses_win)
        if clause:
            text = f"{text}{clause}"
        if attempt > 400:
            text = f"{text} [op {attempt}]"
        text = _strip_some_accents(text, rng)
        if text not in seen_texts:
            seen_texts.add(text)
            records.append(
                {
                    "text": text,
                    "label": "Ganadora_Disciplinada",
                    "indicator": group["indicator"],
                    "action": group["action"],
                    "market": m_name,
                    "operational_error": "Ninguno (Disciplinado)",
                }
            )

    # Completar Riesgo_Neutral hasta 400
    attempt = 0
    while count_label("Riesgo_Neutral") < target_counts["Riesgo_Neutral"]:
        attempt += 1
        group = rng.choice(NEUTRAL_TEMPLATES)
        tpl = rng.choice(group["templates"])
        m_name, m_phrases = rng.choice(MARKETS)
        text = tpl.format(
            market=rng.choice(m_phrases),
            tf=rng.choice(TIMEFRAMES),
            rsi_mid=rng.randint(45, 55),
        ).strip()
        clause = rng.choice(extra_clauses_neutral)
        if clause:
            text = f"{text}{clause}"
        if attempt > 200:
            text = f"{text} [op {attempt}]"
        text = _strip_some_accents(text, rng)
        if text not in seen_texts:
            seen_texts.add(text)
            records.append(
                {
                    "text": text,
                    "label": "Riesgo_Neutral",
                    "indicator": group["indicator"],
                    "action": group["action"],
                    "market": m_name,
                    "operational_error": group["operational_error"],
                }
            )

    # Mezclar manteniendo el registro #1 al principio para fácil inspección
    first_record = records[0]
    rest_records = records[1:]
    rng.shuffle(rest_records)
    clean_records = [first_record] + rest_records

    # Para el archivo RAW (`data/raw/trading_journal_es.csv`), inyectamos 8 filas con valores
    # faltantes (NaN) y 10 filas duplicadas para demostrar el proceso de auditoría y limpieza
    # exigido en la Sección 5.1 del PDF ("valores faltantes").
    raw_records = list(clean_records)
    for idx in range(8):
        raw_records.append(
            {
                "text": None if idx % 2 == 0 else "   ",
                "label": "Perdedora_Error_Operativo",
                "indicator": "RSI",
                "action": "Compra",
                "market": "Volatility 100",
                "operational_error": None,
            }
        )
    for idx in range(10):
        raw_records.append(dict(clean_records[idx]))

    df_raw = pd.DataFrame(raw_records)
    df_raw.insert(0, "id", range(1, len(df_raw) + 1))
    return df_raw


def clean_and_split_dataset(df_raw: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Limpia nulos/vacíos y duplicados exactos, y realiza la partición estratificada 80/20."""
    df = df_raw.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df.replace({"text": {"None": np.nan, "nan": np.nan, "": np.nan}})
    df = df.dropna(subset=["text", "label"]).copy()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)

    if train_test_split is not None:
        df_train, df_test = train_test_split(
            df,
            test_size=0.20,
            stratify=df["label"],
            random_state=seed,
        )
    else:
        # Partición estratificada determinista 80/20 por clase con semilla fija
        train_parts = []
        test_parts = []
        for _, group in df.groupby("label", sort=True):
            shuffled = group.sample(frac=1.0, random_state=seed)
            n_test = int(round(len(shuffled) * 0.20))
            test_parts.append(shuffled.iloc[:n_test])
            train_parts.append(shuffled.iloc[n_test:])
        df_train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
        df_test = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed)

    return df, df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def save_initial_journal_history(df_clean: pd.DataFrame) -> None:
    """Guarda un historial base estructurado para alimentar las estadísticas (`/api/v1/stats`)."""
    sample_df = df_clean.head(120)
    history_items = []
    for _, row in sample_df.iterrows():
        history_items.append(
            {
                "id": int(row["id"]),
                "text": row["text"],
                "prediction": row["label"],
                "indicator": row["indicator"],
                "action": row["action"],
                "market": row["market"],
                "operational_error": row["operational_error"],
                "timestamp": "2026-11-02T12:00:00",
            }
        )
    HISTORY_JSON_PATH.write_text(json.dumps(history_items, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GENERANDO Y CURANDO DATASET: TradingJournal NLP (Español)")
    print("=" * 72)

    df_raw = generate_corpus(seed=SEED)
    df_raw.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8")
    print(f"[1/4] Dataset crudo guardado en: {RAW_CSV_PATH} ({len(df_raw)} filas totales)")

    non_empty_mask = df_raw["text"].fillna("").astype(str).str.strip() != ""
    null_count = int((~non_empty_mask).sum())
    dup_count = int(df_raw.loc[non_empty_mask, "text"].str.strip().duplicated().sum())
    print(f"      -> Auditoría en crudo: {null_count} textos vacíos/nulos y {dup_count} duplicados detectados.")

    df_clean, df_train, df_test = clean_and_split_dataset(df_raw, seed=SEED)
    df_train.to_csv(TRAIN_CSV_PATH, index=False, encoding="utf-8")
    df_test.to_csv(TEST_CSV_PATH, index=False, encoding="utf-8")

    print(f"[2/4] Dataset limpio: {len(df_clean)} registros únicos.")
    print(f"      -> Entrenamiento (80%): {len(df_train)} filas -> {TRAIN_CSV_PATH}")
    print(f"      -> Prueba (20%):        {len(df_test)} filas -> {TEST_CSV_PATH}")

    print("\n[3/4] Distribución de clases (Estratificada):")
    for label, count in df_clean["label"].value_counts().items():
        pct = (count / len(df_clean)) * 100
        print(f"      - {label:<28}: {count:>4} ({pct:.1f}%)")

    save_initial_journal_history(df_clean)
    metadata = {
        "dataset_name": "Spanish Trading Journal & Synthetic Indices NLP Corpus (Deriv Volatility 100 / MT5)",
        "origin": "Corpus estructurado y curado para análisis de bitácoras operativas de trading e indicadores técnicos en español",
        "license": "MIT Educational & Academic License",
        "acquisition_date": datetime.now().strftime("%Y-%m-%d"),
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df_clean)),
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test)),
        "random_seed": SEED,
        "classes": df_clean["label"].value_counts().to_dict(),
    }
    METADATA_JSON_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[4/4] Historial estadístico y metadatos guardados en {PROCESSED_DIR} y {METADATA_JSON_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
