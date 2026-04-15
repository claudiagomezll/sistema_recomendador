import pandas as pd
from bert_score import score
import torch
import gc

def calculate_berts_metrics(reference_text, candidates_dict, lang="es", model_type="bert-base-multilingual-cased"):
    """
    Calcula las métricas BERTScore para un conjunto de candidatos contra una referencia.
    
    Args:
        reference_text (str): El texto de referencia (criterio experto).
        candidates_dict (dict): Diccionario {Nombre_Modelo: Texto_Propuesta}.
        lang (str): Idioma de los textos.
        model_type (str): El modelo de BERT a utilizar.
        
    Returns:
        pd.DataFrame: DataFrame con las métricas (P, R, F1) ordenado por F1.
    """
    if not candidates_dict:
        return pd.DataFrame()

    results = []
    
    # Pre-filtrar textos vacíos
    valid_candidates = {name: text.strip() for name, text in candidates_dict.items() if text and text.strip()}
    
    if not valid_candidates:
        return pd.DataFrame()

    try:
        for model_name, candidate_text in valid_candidates.items():
            # El score espera listas
            P, R, F1 = score(
                [candidate_text],
                [reference_text],
                lang=lang,
                model_type=model_type,
                verbose=False
            )

            results.append({
                "LLM": model_name,
                "Precision": round(float(P.mean()), 4),
                "Recall": round(float(R.mean()), 4),
                "F1": round(float(F1.mean()), 4)
            })
            
            # Limpieza de memoria (importante si se usa en servidores con poca RAM)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        df_results = pd.DataFrame(results)
        # Ordenar por F1 descendente
        df_results = df_results.sort_values(by="F1", ascending=False).reset_index(drop=True)
        return df_results

    except Exception as e:
        print(f"⚠️ Error calculando BERTScore: {e}")
        return pd.DataFrame()

def get_best_model_info(df_results):
    """Extrae el nombre y score del mejor modelo."""
    if df_results.empty:
        return None, 0.0
    
    best_model = df_results.iloc[0]["LLM"]
    best_score = df_results.iloc[0]["F1"]
    return best_model, best_score
