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

    'research_extraction': """Analyze the following user design request and extract the parameters for a scientific serious game proposal.
Input: {query}

Instructions:
1. Infer the sector, context, and learning objectives.
2. Assign a likely Cognitive Function (Perception, Attention, Memory, Inhibitory Control, Working Memory, or Cognitive Flexibility).
3. Assign a target Emotion (anger, disgust, fear, happiness, sadness, or surprise).
4. Assign a VARK style (Visual, Aural, Read/Write, or Kinesthetic).
5. Define necessary Learning Activities and Resources.

Return ONLY a valid JSON object with these keys: 
Sector, Context, Users, Capability, Learning_Objective, Cognitive_Function, Emotion, VARK_Style, Learning_Activities, Learning_Resources, Serious_Game_Type, Motivation, User_Profile.""",

    'synthesis': """*** ATENCIÓN: RESPONDE ÚNICAMENTE EN ESPAÑOL. ***
You are an expert recommendation system for gamified serious games, with knowledge in cognitive science, gamification, and instructional design.

## Scientific Configuration:
- Sector: {Sector} | Context: {Context} | Users: {Users}
- Learning Objective: {Learning_Objective}
- Cognitive Function: {Cognitive_Function}
- Emotion: {Emotion} | VARK Style: {VARK_Style}
- User Profile: {User_Profile}

## Reference Components (from our research):
- Selected MECHANICS: {mechanic}
- Selected ELEMENTS: {element}
- Selected NARRATIVE: {narrative}
- Selected DYNAMICS: {dynamic}

## Instructions:
1. Design a comprehensive serious game proposal based on THESE specific scientific parameters.
2. Use the "Gameplay Bricks Model" (Álvarez) to align mechanics with cognitive functions.
3. The narrative must be immersive, reflect the sector, and trigger the target emotion without trauma.
4. Integrate the Learning Activities and Resources into the mechanics and elements.

## Output Format (Respond in Spanish):
# 🎮 PROYECTO: [Nombre]

### 📊 FICHA TÉCNICA PEDAGÓGICA
- **Objetivo**: {Learning_Objective}
- **Función Cognitiva**: {Cognitive_Function}
- **Estilo VARK**: {VARK_Style}
- **Emoción Diana**: {Emotion}

### 🌍 MUNDO Y NARRATIVA
[Descripción inmersiva del {Context}]

### 🧠 PSICOLOGÍA Y DINÁMICAS
[Cómo se aplican las dinámicas para la {Cognitive_Function}]

### ⚙️ MECÁNICAS E INSTRUCCIÓN
[Detalle del gameplay loop integrando {Learning_Activities}]

### 🕹️ DISEÑO DE INTERFAZ (ELEMENTOS)
[Descripción de los elementos basados en {Learning_Resources}]

*** RECORDATORIO: TODA LA RESPUESTA DEBE ESTAR EN ESPAÑOL. ***"""
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
