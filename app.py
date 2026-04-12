import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from flasgger import Swagger

# Import logic from existing modules
from src.data_loader import load_data_gamification, load_serious_games_ratings
from src.vector_db import initialize_vector_db
from src.algorithms import load_user_ratings_matrix
from src.recommender import RecommenderOrchestrator

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
swagger = Swagger(app)

# Global variables for the orchestrator
ORCHESTRATOR = None

def init_orchestrator():
    """Initializes the recommendation engine components once."""
    global ORCHESTRATOR
    print("🚀 Initializing Recommender Engine...")
    
    # 1. Load data
    df = load_data_gamification()
    if df is None:
        raise Exception("Failed to load dataset.")

    # 2. Initialize Vector DB
    collection = initialize_vector_db(df, dataset_name='serious_games')

    # 3. Load ratings and matrix for FCD
    ratings_df = load_serious_games_ratings()
    matrix, user_map, item_map = load_user_ratings_matrix(ratings_df)

    # 4. Initialize orchestrator
    ORCHESTRATOR = RecommenderOrchestrator(
        df, collection, 
        ratings_df=ratings_df, 
        user_item_matrix=matrix, 
        user_id_map=user_map, 
        item_id_map=item_map
    )
    print("✅ Recommender Engine Initialized.")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ready", "engine": ORCHESTRATOR is not None})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Genera recomendaciones de juegos serios y una propuesta de diseño integrada.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            prompt:
              type: string
              description: Descripción de los objetivos o características del juego deseado.
              example: "juego para enseñar finanzas personales a adolescentes"
            user_id:
              type: integer
              description: ID del usuario para recomendaciones personalizadas.
              example: 1
    responses:
      200:
        description: Análisis completo, componentes ganadores y propuestas narrativas.
        schema:
          type: object
          properties:
            status:
              type: string
            query:
              type: string
            results:
              type: array
              description: Tabla completa de análisis (EDNM).
              items:
                type: object
            synthesis:
              type: object
              description: Componentes ganadores y propuestas finales.
              properties:
                winners:
                  type: object
                proposals:
                  type: object
    """
    if ORCHESTRATOR is None:
        return jsonify({"error": "Engine not initialized"}), 503
        
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400
    
    query = data['prompt']
    user_id = data.get('user_id', 1)  # Default to user 1 if not provided
    
    try:
        # 1. Get categorized recommendations
        recommendations = ORCHESTRATOR.recommend(query, user_id=user_id)
        
        # 2. Generate the synthesis proposal
        synthesis_result = ORCHESTRATOR.generate_proposal(recommendations, query)
        
        return jsonify({
            "status": "success",
            "query": query,
            "results": recommendations, # This is the full list of items with metadata
            "synthesis": synthesis_result  # Contains 'winners' and 'proposals'
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize engine before starting server
    init_orchestrator()
    # Run the Flask app
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
