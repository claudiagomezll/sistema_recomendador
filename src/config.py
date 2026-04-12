import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Paths
DATASET_PATH = BASE_DIR / "data" / "dataset"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

# LLM Configuration
LLMS = {
    "gemini": {
        "enabled": bool(os.getenv("GOOGLE_API_KEY")),
        "provider": "google",
        "model": "gemini-2.5-flash",
        "api_key": os.getenv("GOOGLE_API_KEY"),
    },
    "gpt4o_mini": {
        "enabled": bool(os.getenv("OPENAI_API_KEY")),
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "claude_haiku": {
        "enabled": bool(os.getenv("ANTHROPIC_API_KEY")),
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
}


ACTIVE_LLMS = {k: v for k, v in LLMS.items() if v["enabled"]}

# Recommender Settings
MAX_TOKENS = 1500
TEMPERATURE = 0.7
RETRY_ATTEMPTS = 3
TOP_K_CANDIDATES = 200
VERBOSE = True

# Optimizations
USE_PARALLEL_LLMS = True
USE_RAPIDFUZZ = True
USE_VECTORIZED_MMR = True

# --- Perspective Prompts (EDNM) ---
PROMPT_TEMPLATES = {
    'elements': """You are an expert in gamification ELEMENTS. 
Focus on: badges, points, levels, leaderboards, and UI components.
Candidates:
{context}
Query: {query}
Select the top 3 items that best implement game ELEMENTS for this query.
Return JSON: [{{"recommendation": "Title", "description": "Why (elements focus)", "ranking": 1}}, ...]""",

    'dynamics': """You are an expert in game DYNAMICS. 
Focus on: motivation, player emotions, progression systems, and psychological flow.
Candidates:
{context}
Query: {query}
Select the top 3 items that best implement game DYNAMICS for this query.
Return JSON: [{{"recommendation": "Title", "description": "Why (dynamics focus)", "ranking": 1}}, ...]""",

    'narratives': """You are an expert in game NARRATIVES. 
Focus on: storytelling, characters, world-building, and thematic consistency.
Candidates:
{context}
Query: {query}
Select the top 3 items that best implement game NARRATIVES for this query.
Return JSON: [{{"recommendation": "Title", "description": "Why (narrative focus)", "ranking": 1}}, ...]""",

    'mechanics': """You are an expert in game MECHANICS. 
Focus on: core rules, interaction loops, challenges, and gameplay constraints.
Candidates:
{context}
Query: {query}
Select the top 3 items that best implement game MECHANICS for this query.
Return JSON: [{{"recommendation": "Title", "description": "Why (mechanics focus)", "ranking": 1}}, ...]""",

    'synthesis': """*** ATENCIÓN: RESPONDE ÚNICAMENTE EN ESPAÑOL. EL IDIOMA DE SALIDA DEBE SER 100% ESPAÑOL. ***
Actúa como un Diseñador Maestro de Juegos Serios (Senior Game Designer).
Tu tarea es generar una Propuesta de Diseño Narrativa EXCLUSIVAMENTE EN ESPAÑOL que integre estos 4 pilares:
- ELEMENTO (UI/Ítems): {element}
- DINÁMICA (Psicología/Emoción): {dynamic}
- NARRATIVA (Historia/Mundo): {narrative}
- MECÁNICA (Reglas/Interacción): {mechanic}

Requerimientos del Usuario: {query}

**INSTRUCCIONES CRÍTICAS (SÍGUELAS AL PIE DE LA LETRA):**
1. IDIOMA: ESPAÑOL (PROHIBIDO EL INGLÉS).
2. FORMATO: Markdown limpio (Sin bloques de código JSON).
3. ESTRUCTURA: Usa los encabezados indicados abajo.

**ESTRUCTURA DE LA PROPUESTA (ESCRIBE TODO EN ESPAÑOL):**

# 🎮 PROPUESTA DE JUEGO: [Crea un nombre creativo aquí]

### 🌍 EL MUNDO Y LA ATMÓSFERA
Describe el entorno narrativo y el tono de la historia.

### 🧠 PSICOLOGÍA DEL FLUJO (DINÁMICAS)
Explica cómo las DINÁMICAS mencionadas crean la experiencia emocional.

### 🛠️ LAS HERRAMIENTAS DE SUPERVIVENCIA (ELEMENTOS)
Describe cómo los ELEMENTOS aparecen en el mundo.

### ⚙️ EL RITMO DEL JUEGO (MECÁNICAS)
Explica el ciclo de juego integrando las MECÁNICAS.

### 🕹️ "UN DÍA EN LA PARTIDA" (NARRATIVA EN ACCIÓN)
Una descripción vívida de un momento del juego.

*** RECORDATORIO FINAL: TODA LA RESPUESTA DEBE ESTAR EN ESPAÑOL. ***"""
}

# Configuration Constants
DATASETS = {
    'serious_games': 'data/gamification_dataset.csv'
}



# Dataset Config
DATASETS = {
    'serious_games': {
        'columns': {
            'id': 'item_id',
            'title': 'title',
            'description': 'description'
        },
        'embedding_fields': ['title', 'description', 'item_type', 'type_label'],
        'top_k': 12,
        'threshold': 0.55,
        'mmr_lambda': 0.75,
    }
}
