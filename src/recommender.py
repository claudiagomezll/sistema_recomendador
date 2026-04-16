import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import (
    ACTIVE_LLMS, VERBOSE, TOP_K_CANDIDATES, DATASETS, PROMPT_TEMPLATES,
    EVALUATE_QUALITY, BERT_SCORE_MODEL, BERT_SCORE_LANG
)
from .llm_clients import create_llm_provider
from .vector_db import rag_retrieval, get_encoder
from .algorithms import (
    calculate_weights, fcd_recommendations, fuzzy_match_title,
    calculate_ranking_rrf, mmr_diversification
)
from .evaluation import calculate_berts_metrics, get_best_model_info

class RecommenderOrchestrator:
    def __init__(self, df, collection, ratings_df=None, user_item_matrix=None, user_id_map=None, item_id_map=None):
        self.df = df
        self.collection = collection
        self.ratings_df = ratings_df
        self.user_item_matrix = user_item_matrix
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.encoder = get_encoder()
        self.current_research_context = None

    def set_research_context(self, research_data):
        """Manually sets the research context from structured expert input."""
        if VERBOSE: print("🧪 Cargando contexto de investigación manual (Modo Experto)...")
        # Ensure keys match expected template placeholders
        self.current_research_context = research_data
        return self.current_research_context

    def extract_research_parameters(self, query):
        if VERBOSE: print("🧬 Extrayendo parámetros científicos de investigación...")
        prompt = PROMPT_TEMPLATES['research_extraction'].format(query=query)
        
        # Use a reliable LLM for extraction (e.g. gpt4o_mini or the first active one)
        llm_name = list(ACTIVE_LLMS.keys())[0]
        cfg = ACTIVE_LLMS[llm_name]
        provider = create_llm_provider(cfg['provider'], cfg['model'], api_key=cfg['api_key'])
        
        try:
            raw_response = provider.generate(prompt)
            import json
            import re
            # Extract JSON block
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                self.current_research_context = json.loads(json_match.group(0))
            else:
                self.current_research_context = json.loads(raw_response)
            
            if VERBOSE: print(f"   ✅ Parámetros extraídos: {self.current_research_context.get('Cognitive_Function', 'N/A')} | {self.current_research_context.get('VARK_Style', 'N/A')}")
        except Exception as e:
            if VERBOSE: print(f"   ⚠️ Error extrayendo parámetros: {e}")
            self.current_research_context = {
                "Sector": "General", "Context": "Gaming", "Users": "Players",
                "Learning_Objective": query, "Cognitive_Function": "General",
                "Emotion": "Engagement", "VARK_Style": "Multi-modal"
            }
        return self.current_research_context

    def query_all_llms(self, query, candidates, perspective=None, json_mode=True):
        if perspective and perspective in PROMPT_TEMPLATES:
            # Check if it's the synthesis prompt which needs special placeholders
            if perspective == 'synthesis' and isinstance(candidates, dict):
                prompt = PROMPT_TEMPLATES['synthesis'].format(
                    query=query,
                    element=candidates.get('element', 'N/A'),
                    dynamic=candidates.get('dynamic', 'N/A'),
                    narrative=candidates.get('narrative', 'N/A'),
                    mechanic=candidates.get('mechanic', 'N/A')
                )
            else:
                # Build context for standard perspective prompts
                context = ""
                # Build context from candidates if provided
                if hasattr(candidates, 'iterrows'):
                    for _, row in candidates.iterrows():
                        context += f"- ID: {row['id']}, Título: {row['title']}, Descripción: {row['description']}\n"
                    prompt = PROMPT_TEMPLATES[perspective].format(query=query, context=context)
                else:
                    prompt = PROMPT_TEMPLATES[perspective].format(query=query, context=context)
        else:
            # Fallback to default simple prompt
            prompt = f"Query: {query}\nCandidates:\n"
            if hasattr(candidates, 'iterrows'):
                for _, row in candidates.iterrows():
                    prompt += f"- {row['title']}: {row['description']}\n"
            prompt += "\nSelect the top 5 most relevant items and explain why. Format as JSON: [{\"recommendation\": \"Title\", \"description\": \"Why\", \"ranking\": 1}, ...]"

        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    create_llm_provider(cfg['provider'], cfg['model'], api_key=cfg['api_key']).generate, 
                    prompt, 
                    json_mode=json_mode
                ): name for name, cfg in ACTIVE_LLMS.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    import json
                    import re
                    raw_result = future.result()
                    
                    if not json_mode:
                        results[name] = raw_result
                        continue

                    # More robust JSON extraction
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', raw_result, re.DOTALL)
                    if json_match:
                        parsed_result = json.loads(json_match.group(0))
                    else:
                        # Fallback to plain loads
                        parsed_result = json.loads(raw_result)
                    
                    # Extract list if wrapped in a dict
                    if isinstance(parsed_result, dict):
                        for key in ['recommendations', 'items', 'results', 'data']:
                            if key in parsed_result and isinstance(parsed_result[key], list):
                                parsed_result = parsed_result[key]
                                break
                    
                    if isinstance(parsed_result, list):
                        results[name] = parsed_result
                        if VERBOSE: print(f"   ✅ LLM {name} entregó {len(parsed_result)} recomendaciones.")
                    else:
                        if VERBOSE: print(f"   ⚠️ Warning LLM {name}: Expected list, got {type(parsed_result).__name__}")
                except Exception as e:
                    if VERBOSE: print(f"   ⚠️ Error LLM {name} (Parsing): {e}")
        return results

    def semantic_validation(self, llm_results, query, target_item_type=None):
        query_emb = self.encoder.encode([query])[0]
        validated = []
        
        for llm_name, recs in llm_results.items():
            if not isinstance(recs, list):
                if VERBOSE: print(f"⚠️ Skipping LLM {llm_name} results: not a list")
                continue
            for rec in recs:
                if not isinstance(rec, dict) or 'recommendation' not in rec:
                    continue
                title = rec['recommendation']
                # Search in DB
                matched_title, score, item_data = fuzzy_match_title(title, self.df)
                
                if item_data is not None:
                    # Flexible type matching (handles plurals like 'elements' vs 'element')
                    if target_item_type:
                        db_type = str(item_data['item_type']).lower()
                        target_type = target_item_type.lower().rstrip('s')
                        if db_type != target_type and db_type != target_item_type.lower():
                            continue
                    
                    item_text = f"{item_data['title']}. {item_data['description']}"
                    item_emb = self.encoder.encode([item_text])[0]
                    sim = np.dot(query_emb, item_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(item_emb) + 1e-10)
                    sim_norm = float((sim + 1) / 2)
                    
                    if sim_norm >= 0.55: # Threshold
                        valid_item = {
                            'llm_source': llm_name,
                            'perspective': target_item_type or 'general', # Reusing target_item_type parameter as perspective label
                            'title': item_data['title'],
                            'item_id': item_data['item_id'],
                            'item_type': item_data['item_type'],
                            'description': item_data['description'],
                            'relevance_score': sim_norm,
                            'llm_reasoning': rec['description']
                        }
                        validated.append(valid_item)
                        if VERBOSE: 
                            print(f"      ✅ Validado: {valid_item['title']} (Score: {sim_norm:.3f})")
        return validated

    def recommend(self, query, user_id=None, top_k=3, research_data=None):
        """
        Calcula recomendaciones híbridas integrando RAG y Filtrado Colaborativo.
        Si se provee research_data, se usa para el contexto científico.
        """
        if research_data:
            self.set_research_context(research_data)
            # Use specific fields from research_data to improve search
            search_query = f"{research_data.get('Learning_Objective', '')} {research_data.get('Cognitive_Function', '')} {query}".strip()
        else:
            self.extract_research_parameters(query)
            search_query = query
        
        if VERBOSE:
            print(f"🚀 Iniciando recomendación multi-perspectiva (EDNM) para: '{query}'")
            if research_data: print(f"   🧬 Modo Experto activado: Objetivo '{self.current_research_context.get('Learning_Objective')}'")

        # 1. Weights
        weights = calculate_weights(user_id, self.ratings_df)
        
        # Perspectives matching the database labels (element, dynamic, narrative, mechanic)
        perspectives = ['element', 'dynamic', 'narrative', 'mechanic']
        all_validated = []
        
        # 3. Executes each perspective with filtered retrieval
        for p in perspectives:
            if VERBOSE: print(f"   🔍 Procesando perspectiva: {p.upper()}...")
            
            # Filtered retrieval using the enriched search_query
            candidates = rag_retrieval(search_query, self.collection, top_k=20, where={"item_type": p})
            
            if candidates.empty:
                if VERBOSE: print(f"      ⚠️ No se encontraron candidatos de tipo '{p}'")
                continue
                
            # We add 's' for the prompt context if needed, but the label 'p' is used for validation
            llm_results = self.query_all_llms(query, candidates, perspective=p+'s' if not p.endswith('s') else p)
            validated = self.semantic_validation(llm_results, query, target_item_type=p)
            
            # Sort by relevance and take top 3 for this perspective
            validated.sort(key=lambda x: x['relevance_score'], reverse=True)
            all_validated.extend(validated[:3]) 

        # 4. FCD Recommendations (Optional/Social)
        fcd_results = []
        if user_id and self.user_item_matrix is not None:
            fcd_results = fcd_recommendations(user_id, self.ratings_df, self.user_item_matrix, 
                                              self.user_id_map, self.item_id_map)
            for f in fcd_results[:3]: # Take top 3 from FCD too
                item = self.df[self.df['global_item_id'] == f['item_id']]
                if not item.empty:
                    row = item.iloc[0]
                    all_validated.append({
                        'llm_source': 'fcd',
                        'perspective': 'social',
                        'title': row['title'],
                        'item_id': row['item_id'],
                        'item_type': row['item_type'],
                        'description': row['description'],
                        'relevance_score': 0.8,
                        'fcd_score': f['fcd_score'],
                        'llm_reasoning': "Recommended by similar users profile."
                    })

        # 5. Final Ranking & Strict Grouping by Category
        ranked = calculate_ranking_rrf(all_validated, weights)
        
        # Enforce grouping: Dynamic -> Element -> Mechanic -> Narrative
        category_priority = {'dynamic': 0, 'element': 1, 'mechanic': 2, 'narrative': 3, 'social': 4}
        # We sort a copy to be safe and ensure it's a list
        final_ranked = sorted(
            ranked, 
            key=lambda x: (category_priority.get(x.get('perspective'), 99), -x.get('rank_score', 0))
        )
        
        if VERBOSE:
            print(f"✅ Resultados finales agrupados: {len(final_ranked)} ítems.")
            
        return final_ranked
    def _format_winner_for_prompt(self, val):
        """Convierte un resultado JSON de perspectiva en texto legible para el prompt de síntesis."""
        if not val or val == 'N/A' or val == '...':
            return "No disponible"
        try:
            import json
            # Intentar limpiar si viene con markdown
            val_clean = str(val).replace('```json', '').replace('```', '').strip()
            if (val_clean.startswith('[') and val_clean.endswith(']')) or (val_clean.startswith('{') and val_clean.endswith('}')):
                data = json.loads(val_clean)
                if isinstance(data, list):
                    items = [item.get('recommendation', item.get('title', '')) for item in data]
                    return ", ".join([i for i in items if i])
                elif isinstance(data, dict):
                    return data.get('recommendation', data.get('title', str(data)))
        except:
            pass
        return str(val)

    def generate_proposal(self, ranked_results, query):
        if VERBOSE: print("\n🎨 Generando propuesta de diseño integrada...")
        
        # 1. Extract best of each category
        selected_winners = {}
        perspectives = ['element', 'dynamic', 'narrative', 'mechanic']
        
        for p in perspectives:
            best = next((x for x in ranked_results if x.get('perspective') == p), None)
            if best:
                selected_winners[p] = f"{best['title']} - {best['llm_reasoning']}"
                if VERBOSE: print(f"   🏆 Ganador {p.upper()}: {best['title']}")

        if not selected_winners:
            return {"error": "No hay suficientes componentes validados para generar una propuesta."}

        # 2. Extract research context if not present
        if not hasattr(self, 'current_research_context') or not self.current_research_context:
            self.extract_research_parameters(query)

        ctx = self.current_research_context
        
        # 3. Build the specific research master prompt
        master_prompt = PROMPT_TEMPLATES['research_master_template'].format(
            Sector=ctx.get('Sector', 'General'),
            Context=ctx.get('Context', 'N/A'),
            Serious_Game_Type=ctx.get('Serious_Game_Type', 'Adventure'),
            Users=ctx.get('Users', 'Players'),
            Learning_Objective=ctx.get('Learning_Objective', query),
            Cognitive_Function=ctx.get('Cognitive_Function', 'General'),
            Capability=ctx.get('Capability', 'General'),
            VARK_Style=ctx.get('VARK_Style', 'Visual/Aural'),
            Emotion=ctx.get('Emotion', 'Engagement'),
            Motivation=ctx.get('Motivation', 'Intrinsic'),
            Learning_Activities=ctx.get('Learning_Activities', 'Gameplay'),
            Learning_Resources=ctx.get('Learning_Resources', 'Game elements'),
            Basic_Mechanics=ctx.get('Basic_Mechanics', 'Interaction')
        )
        
        synthesis_prompt = PROMPT_TEMPLATES['synthesis'].format(
            master_prompt=master_prompt,
            element=self._format_winner_for_prompt(selected_winners.get('element', 'N/A')),
            dynamic=self._format_winner_for_prompt(selected_winners.get('dynamic', 'N/A')),
            narrative=self._format_winner_for_prompt(selected_winners.get('narrative', 'N/A')),
            mechanic=self._format_winner_for_prompt(selected_winners.get('mechanic', 'N/A')),
            Context=ctx.get('Context', 'Gaming'),
            Cognitive_Function=ctx.get('Cognitive_Function', 'General'),
            Sector=ctx.get('Sector', 'General'),
            Users=ctx.get('Users', 'Players'),
            Learning_Objective=ctx.get('Learning_Objective', query),
            Emotion=ctx.get('Emotion', 'Engagement'),
            VARK_Style=ctx.get('VARK_Style', 'Multi-modal')
        )

        raw_proposals = self.query_all_llms(synthesis_prompt, [], json_mode=False)
        
        clean_proposals = {}
        import re
        import json
        for name, text in raw_proposals.items():
            if not text: 
                clean_proposals[name] = "Error: Sin respuesta del modelo."
                continue
            
            # 1. Quitar etiquetas de bloque de código markdown redundantes
            text = re.sub(r'```[a-zA-Z]*\n', '', text)
            text = text.replace('```', '').strip()
            
            # 2. Extractor de JSON robusto (busca el bloque más grande entre [] o {})
            json_pattern = r'(\[[\s\S]*\]|\{[\s\S]*\})'
            match = re.search(json_pattern, text)
            
            if match:
                potential_json = match.group(0)
                try:
                    parsed = json.loads(potential_json)
                    new_text = ""
                    # Caso A: Lista de recomendaciones
                    if isinstance(parsed, list):
                        new_text = "# 🎮 PROPUESTA DE DISEÑO ESTRATÉGICO\n\n"
                        for item in parsed:
                            if isinstance(item, dict):
                                head = item.get('recommendation', item.get('title', item.get('titulo', item.get('heading', item.get('recomendacion', '')))))
                                body = item.get('description', item.get('content', item.get('descripcion', item.get('contenido', item.get('body', '')))))
                                if not head and item.values():
                                    head = list(item.values())[0] if not head else head
                                    if len(item.values()) > 1 and not body:
                                        body = list(item.values())[1]
                                new_text += f"### {head}\n{body}\n\n"
                            else:
                                new_text += f"{item}\n\n"
                    elif isinstance(parsed, dict):
                        title = parsed.get('title', parsed.get('propuesta', parsed.get('titulo', 'PROPUESTA DE JUEGO')))
                        new_text = f"# 🎮 {title}\n\n"
                        sections = parsed.get('sections', [])
                        if not sections and 'secciones' in parsed: sections = parsed['secciones']
                        if isinstance(sections, list):
                            for s in sections:
                                if isinstance(s, dict):
                                    head = s.get('heading', s.get('title', s.get('header', s.get('titulo', s.get('nombre', 'Sección')))))
                                    body = s.get('content', s.get('body', s.get('contenido', '')))
                                    new_text += f"### {head}\n{body}\n\n"
                                else:
                                    new_text += f"{s}\n\n"
                        else:
                            for k, v in parsed.items():
                                if k.lower() not in ['title', 'sections', 'propuesta', 'titulo', 'secciones']:
                                    new_text += f"### {k.replace('_', ' ').upper()}\n{v}\n\n"
                    if new_text:
                        # Reemplazo in-place para mantener el texto narrativo alrededor
                        text = text.replace(potential_json, "\n" + new_text)
                except:
                    pass
            clean_proposals[name] = text

        # 5. BERTScore Quality Evaluation
        quality_metrics = None
        best_model_data = None
        
        if EVALUATE_QUALITY:
            if VERBOSE: print("\n📏 Calculando métricas de calidad (BERTScore)...")
            
            # Generar el texto de referencia "experto" dinámicamente
            reference_text = f"""
            Título del juego: Propuesta de Juego Serio para {ctx.get('Sector', 'General')}.
            El juego serio gamificado debe permitir a los usuarios {ctx.get('Users', 'Players')} 
            lograr el objetivo de {ctx.get('Learning_Objective', query)} y entrenarse en la 
            función cognitiva {ctx.get('Cognitive_Function', 'General')} para mejorar la 
            capacidad {ctx.get('Capability', 'General')}. 
            Debe incluir mecánicas de {ctx.get('Basic_Mechanics', 'Interacción')} y estar 
            alineado con la emoción {ctx.get('Emotion', 'Engagement')}.
            """
            
            quality_metrics_df = calculate_berts_metrics(
                reference_text, 
                clean_proposals, 
                lang=BERT_SCORE_LANG, 
                model_type=BERT_SCORE_MODEL
            )
            
            if not quality_metrics_df.empty:
                quality_metrics = quality_metrics_df.to_dict(orient='records')
                best_name, best_f1 = get_best_model_info(quality_metrics_df)
                best_model_data = {"model": best_name, "f1_score": best_f1}
                if VERBOSE: print(f"   🏆 Mejor salida según BERTScore-F1: {best_name} ({best_f1:.4f})")
            
        return {
            "winners": selected_winners,
            "proposals": clean_proposals,
            "quality_metrics": quality_metrics,
            "best_proposal": best_model_data,
            "research_context": self.current_research_context
        }
