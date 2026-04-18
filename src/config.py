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
EVALUATE_QUALITY = True  # BERTScore evaluation toggle
BERT_SCORE_MODEL = "bert-base-multilingual-cased"
BERT_SCORE_LANG = "es"

# --- Perspective Prompts (EDNM) ---
PROMPT_TEMPLATES = {
    'research_extraction': """Analiza la siguiente consulta y extrae los parámetros de investigación científica para el diseño de un juego serio. 
Retorna UNICAMENTE un objeto JSON con los siguientes campos:
- Sector (Sector industrial o social)
- Context (Contexto específico de aplicación)
- Serious_Game_Type (Tipo de juego: entrenamiento, educativo, simulador, etc.)
- Users (Perfil de los usuarios destinatarios)
- Learning_Objective (Objetivo de aprendizaje principal)
- Cognitive_Function (Función cognitiva: Memoria, Percepción, Atención, etc.)
- Capability (Capacidad específica a mejorar)
- VARK_Style (Estilo de aprendizaje: Visual, Aural, Read/Write, Kinesthetic)
- Emotion (Emoción objetivo: Engagement, Challenge, Curiosity, etc.)
- Motivation (Tipo de motivación: Intrínseca, Extrínseca)
- Learning_Activities (Actividades pedagógicas)
- Learning_Resources (Recursos necesarios)
- Basic_Mechanics (Mecánicas básicas)

Consulta: {query}
JSON:""",

    'elements': """Role: Act as an expert recommendation system for gamified serious games, specializing in the {Sector} sector and creating immersive learning experiences for high-risk environments.

Recommend a gamified serious game for the {Sector} in the {Context}. The serious game must enable users to achieve the {Learning_Objective}, and train the {Cognitive_Function} with the goal of improving {Capability}. This must be considered alongside the learning style {VARK_Style}, the emotion {Emotion}, and the predominant motivation {Motivation}. The learning activities {Learning_Activities} and resources {Learning_Resources} of the serious game are designed in accordance with the learning style {VARK_Style}. The mechanics {Basic_Mechanics} of the serious game are included using the “Gameplay Bricks Model, a Theoretical Framework to Match Game Mechanics and Cognitive Functions” proposed by Álvarez.
Using the input elements and certain considerations, the output must include dynamics, mechanics, elements, and a gamified narrative consistent with the cognitive function and emotion.

Specific Focus - 3. Gamification Elements: These consider the {Learning_Resources} plus additional gamification resources. Detail these elements and how they support the user profile and game objectives.

Candidates:
{context}

Query: {query}

Select the top 3 items that best implement game ELEMENTS for this query.
Return JSON: [{{"recommendation": "Title", "description": "3. Gamification Elements: [Detailed explanation of resources and support for user profile]", "ranking": 1}}, ...]""",

    'dynamics': """Role: Act as an expert recommendation system for gamified serious games, specializing in the {Sector} sector and creating immersive learning experiences for high-risk environments.

Recommend a gamified serious game for the {Sector} in the {Context}. The serious game must enable users to achieve the {Learning_Objective}, and train the {Cognitive_Function} with the goal of improving {Capability}. This must be considered alongside the learning style {VARK_Style}, the emotion {Emotion}, and the predominant motivation {Motivation}. The learning activities {Learning_Activities} and resources {Learning_Resources} of the serious game are designed in accordance with the learning style {VARK_Style}. The mechanics {Basic_Mechanics} of the serious game are included using the “Gameplay Bricks Model, a Theoretical Framework to Match Game Mechanics and Cognitive Functions” proposed by Álvarez.
Using the input elements and certain considerations, the output must include dynamics, mechanics, elements, and a gamified narrative consistent with the cognitive function and emotion.

Specific Focus - 1. Gamified Dynamics: These must take into account the context, basic mechanics, and user profile to determine the following outputs:
- Progression Dynamics: The user must {Basic_Mechanics} to achieve the {Learning_Objective}.
- Emotional Dynamics: The user must experience the {Emotion}.
- Cognitive Function Dynamics: The user must improve the capacity of {Capability} to maintain the {Cognitive_Function}.
- Motivational Dynamics: The user must perform a {Basic_Mechanics} action to fulfill the motivation {Motivation}.

Candidates:
{context}

Query: {query}

Select the top 3 items that best implement game DYNAMICS for this query.
Return JSON: [{{"recommendation": "Title", "description": "1. Gamified Dynamics: [Detailed explanation of Progression, Emotional, Cognitive, and Motivational components]", "ranking": 1}}, ...]""",

    'narratives': """Role: Act as an expert recommendation system for gamified serious games, specializing in the {Sector} sector and creating immersive learning experiences for high-risk environments.

Recommend a gamified serious game for the {Sector} in the {Context}. The serious game must enable users to achieve the {Learning_Objective}, and train the {Cognitive_Function} with the goal of improving {Capability}. This must be considered alongside the learning style {VARK_Style}, the emotion {Emotion}, and the predominant motivation {Motivation}. The learning activities {Learning_Activities} and resources {Learning_Resources} of the serious game are designed in accordance with the learning style {VARK_Style}. The mechanics {Basic_Mechanics} of the serious game are included using the “Gameplay Bricks Model, a Theoretical Framework to Match Game Mechanics and Cognitive Functions” proposed by Álvarez.
Using the input elements and certain considerations, the output must include dynamics, mechanics, elements, and a gamified narrative consistent with the cognitive function and emotion.

Specific Focus - 4. Narrative of the Gamified Serious Game: This considers the {Context} and the gamified dynamics to achieve the learning objective. The narrative should be coherent, immersive, and facilitate the detection of warning signs (as an outcome of improved attention), decision-making under pressure, and understanding of the identified deficiencies.

Candidates:
{context}

Query: {query}

Select the top 3 items that best implement game NARRATIVES for this query.
Return JSON: [{{"recommendation": "Title", "description": "4. Narrative: [Detailed explanation of coherence, immersion, and gameplay implications]", "ranking": 1}}, ...]""",

    'mechanics': """Role: Act as an expert recommendation system for gamified serious games, specializing in the {Sector} sector and creating immersive learning experiences for high-risk environments.

Recommend a gamified serious game for the {Sector} in the {Context}. The serious game must enable users to achieve the {Learning_Objective}, and train the {Cognitive_Function} with the goal of improving {Capability}. This must be considered alongside the learning style {VARK_Style}, the emotion {Emotion}, and the predominant motivation {Motivation}. The learning activities {Learning_Activities} and resources {Learning_Resources} of the serious game are designed in accordance with the learning style {VARK_Style}. The mechanics {Basic_Mechanics} of the serious game are included using the “Gameplay Bricks Model, a Theoretical Framework to Match Game Mechanics and Cognitive Functions” proposed by Álvarez.
Using the input elements and certain considerations, the output must include dynamics, mechanics, elements, and a gamified narrative consistent with the cognitive function and emotion.

Specific Focus - 2. Gamified Mechanics: These are considered the combination of each of the {Learning_Activities} plus the {Learning_Resources}. Explain how the basic mechanics ({Basic_Mechanics}) are integrated with these activities and resources, referencing the "Gameplay Bricks Model" and focusing on developing {Cognitive_Function}.

Candidates:
{context}

Query: {query}

Select the top 3 items that best implement game MECHANICS for this query.
Return JSON: [{{"recommendation": "Title", "description": "2. Gamified Mechanics: [Detailed explanation of activities + resources + bricks model integration]", "ranking": 1}}, ...]""",

    'research_master_template': """Se requiere recomendar un juego serio gamificado para el sector {Sector} en el contexto {Context}. El juego serio {Serious_Game_Type} debe permitir a los usuarios {Users} lograr el objetivo de aprendizaje {Learning_Objective} y entrenarse en la función cognitiva {Cognitive_Function} para mejorar la capacidad {Capability} considerando el estilo de aprendizaje {VARK_Style}, la emoción {Emotion} y la motivación predominante {Motivation}. Las actividades {Learning_Activities} y recursos {Learning_Resources} del juego serio {Serious_Game_Type} se consideran de acuerdo con el estilo de aprendizaje {VARK_Style}. Las mecánicas {Basic_Mechanics} del juego serio {Serious_Game_Type} son incluidas empleando el marco de trabajo “Gameplay Bricks Model, a Theoretical Framework to Match Game Mechanics and Cognitive Functions” propuesto por Álvarez. Empleando los items de entrada y algunos valores de referencia se debe asegurar que la salida incluya dinámicas, mecánicas, elementos y una narrativa gamificada coherente con la función cognitiva y la emoción.""",

    'synthesis': """*** ATENCIÓN: RESPONDE ÚNICAMENTE EN ESPAÑOL. ***
{master_prompt}

## Componentes de Referencia (Validados de la Base de Datos):
- DINÁMICA: {dynamic}
- ELEMENTO: {element}
- MECÁNICA: {mechanic}
- NARRATIVA: {narrative}

## Instrucciones de Diseño:
1. Diseña la propuesta siguiendo el marco de trabajo "Gameplay Bricks Model" para alinear mecánicas con la función {Cognitive_Function}.
2. La narrativa DEBE ser inmersiva y reflejar el contexto: {Context}.
3. Asegura la coherencia entre todos los niveles de EDNM.

## Formato de Salida (Markdown):
# 🎮 PROYECTO: [Nombre Sugerido]

### 🧪 ANÁLISIS CIENTÍFICO (Items de Entrada)
- **Sector**: {Sector} | **Usuarios**: {Users}
- **Objetivo**: {Learning_Objective}
- **Función Cognitiva**: {Cognitive_Function}
- **Emoción/Estilo**: {Emotion} / {VARK_Style}

### 🌍 NARRATIVA INMERSIVA
[Desarrollo histórico/mundano coherente con el contexto]

### ⚙️ ESTRUCTURA EDNM (Gameplay Bricks)
- **Dinámicas**: [Explicación psicológica]
- **Mecánicas**: [Reglas y actividades de aprendizaje]
- **Elementos**: [Componentes de interfaz y recursos]

### 🕹️ "EL MOMENTO DE LA VERDAD"
[Descripción en primera persona de la experiencia del usuario]"""
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
