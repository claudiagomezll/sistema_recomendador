import sys
import os
import json
import pandas as pd
from src.config import DATASETS, VERBOSE
from src.data_loader import load_data_gamification, load_serious_games_ratings
from src.vector_db import initialize_vector_db
from src.algorithms import load_user_ratings_matrix
from src.recommender import RecommenderOrchestrator

def main():
    print("="*50)
    print("SISTEMA DE RECOMENDACIÓN MULTI-LLM E HÍBRIDO")
    print("="*50)

    # 1. Cargar datos
    df = load_data_gamification()
    if df is None:
        print("❌ Error: No se pudo cargar el dataset.")
        return

    # 2. Inicializar Vector DB
    collection = initialize_vector_db(df, dataset_name='serious_games')

    # 3. Cargar ratings y matriz para FCD
    ratings_df = load_serious_games_ratings()
    matrix, user_map, item_map = load_user_ratings_matrix(ratings_df)

    # 4. Inicializar orquestador
    orchestrator = RecommenderOrchestrator(
        df, collection, 
        ratings_df=ratings_df, 
        user_item_matrix=matrix, 
        user_id_map=user_map, 
        item_id_map=item_map
    )

    # 5. Ejecutar recomendación de ejemplo
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "juego para mejorar habilidades de toma de decisiones bajo presión y trabajo en equipo"

    user_id = 1 # Usuario de ejemplo
    
    print(f"\nProcesando recomendación para el usuario {user_id}...")
    final_recommendations = orchestrator.recommend(query, user_id=user_id, top_k=5)

    # 6. Mostrar RESULTADOS COMPLETOS en formato TABLA
    print("\n" + "="*80)
    print("1. TABLA COMPLETA DE ANÁLISIS (EDNM)")
    print("="*80)
    
    if final_recommendations:
        results_df = pd.DataFrame(final_recommendations)
        
        # Tabla completa
        full_display_df = results_df[['perspective', 'title', 'rank_score', 'llm_reasoning']].copy()
        full_display_df.columns = ['Perspectiva', 'Título', 'Score', 'Análisis']
        full_display_df['Análisis'] = full_display_df['Análisis'].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
        print(full_display_df.to_string(index=False))
        
        # 7. Generar Propuesta de Juego (Síntesis)
        synthesis_result = orchestrator.generate_proposal(final_recommendations, query)
        
        if "error" in synthesis_result:
            print(f"\n❌ Error en la síntesis: {synthesis_result['error']}")
            return

        # --- MOSTRAR COMPONENTES GANADORES ---
        print("\n" + "="*80)
        print("2. COMPONENTES GANADORES SELECCIONADOS")
        print("="*80)
        for cat, desc in synthesis_result['winners'].items():
            print(f"🔸 {cat.upper()}: {desc}")
        
        # --- MOSTRAR PROPUESTAS ---
        print("\n" + "="*80)
        print("3. PROPUESTA DE DISEÑO INTEGRADA (SÍNTESIS)")
        print("="*80)
        
        for model_name, proposal in synthesis_result['proposals'].items():
            print(f"\n💡 PROPUESTA DE: {model_name.upper()}")
            print("-" * 40)
            print(proposal)
            print("-" * 80)
    else:
        print("❌ No se encontraron recomendaciones válidas para esta consulta.")
    
    print("\n✅ Proceso completado con éxito.")

if __name__ == "__main__":
    main()
