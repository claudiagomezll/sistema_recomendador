import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import ACTIVE_LLMS, VERBOSE, TOP_K_CANDIDATES, DATASETS, PROMPT_TEMPLATES
from .llm_clients import create_llm_provider
from .vector_db import rag_retrieval, get_encoder
from .algorithms import (
    calculate_weights, fcd_recommendations, fuzzy_match_title,
    calculate_ranking_rrf, mmr_diversification
)

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
                    prompt
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

    def recommend(self, query, user_id=None, top_k=3):
        if VERBOSE: print(f"🚀 Iniciando recomendación multi-perspectiva (EDNM) para: '{query}'")
        
        # 1. Weights
        weights = calculate_weights(user_id, self.ratings_df)
        
        # Perspectives matching the database labels (element, dynamic, narrative, mechanic)
        perspectives = ['element', 'dynamic', 'narrative', 'mechanic']
        all_validated = []
        
        # 3. Executes each perspective with filtered retrieval
        for p in perspectives:
            if VERBOSE: print(f"   🔍 Procesando perspectiva: {p.upper()}...")
            
            # Filtered retrieval: only items of type 'p'
            candidates = rag_retrieval(query, self.collection, top_k=20, where={"item_type": p})
            
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

    def generate_proposal(self, ranked_results, query):
        if VERBOSE: print("\n🎨 Generando propuesta de diseño integrada...")
        
        # Extract best of each category
        winners = {}
        perspectives = ['element', 'dynamic', 'narrative', 'mechanic']
        
        for p in perspectives:
            # Find the best item for this perspective
            best = next((x for x in ranked_results if x.get('perspective') == p), None)
            if best:
                winners[p] = f"{best['title']} - {best['llm_reasoning']}"
                if VERBOSE: print(f"   🏆 Ganador {p.upper()}: {best['title']}")

        if not winners:
            return {"error": "No hay suficientes componentes validados para generar una propuesta."}

        # 4. Generate synthesis with all configured models
        # Ensure we have defaults if a category was missed
        
        # Extract research context if not present
        if not hasattr(self, 'current_research_context') or not self.current_research_context:
            self.extract_research_parameters(query)

        ctx = self.current_research_context
        synthesis_prompt = PROMPT_TEMPLATES['synthesis'].format(
            element=winners.get('element', 'N/A'),
            dynamic=winners.get('dynamic', 'N/A'),
            narrative=winners.get('narrative', 'N/A'),
            mechanic=winners.get('mechanic', 'N/A'),
            query=query,
            Sector=ctx.get('Sector', 'General'),
            Context=ctx.get('Context', 'Gaming'),
            Users=ctx.get('Users', 'Players'),
            Capability=ctx.get('Capability', 'General'),
            Learning_Objective=ctx.get('Learning_Objective', query),
            Cognitive_Function=ctx.get('Cognitive_Function', 'General'),
            Emotion=ctx.get('Emotion', 'Engagement'),
            VARK_Style=ctx.get('VARK_Style', 'Multi-modal'),
            Learning_Activities=ctx.get('Learning_Activities', 'Gameplay'),
            Learning_Resources=ctx.get('Learning_Resources', 'Game elements'),
            Serious_Game_Type=ctx.get('Serious_Game_Type', 'Educational'),
            Motivation=ctx.get('Motivation', 'Intrinsic'),
            User_Profile=ctx.get('User_Profile', 'Standard')
        )
        
        raw_proposals = self.query_all_llms(synthesis_prompt, [], json_mode=False)
        
        # Clean and standardize proposals (detecting accidental JSON or markdown blocks)
        clean_proposals = {}
        for model_name, text in raw_proposals.items():
            if not isinstance(text, str):
                text = str(text)
            
            # 1. Strip markdown code blocks
            import re
            text = re.sub(r'```[a-zA-Z]*\n', '', text)
            text = text.replace('```', '')
            text = text.strip()
            
            # 2. Auto-fix JSON-formatted responses (common in models like Gemini)
            if (text.startswith('[') and text.endswith(']')) or (text.startswith('{') and text.endswith('}')):
                import json
                try:
                    parsed = json.loads(text)
                    new_text = ""
                    # Case A: List of dicts
                    if isinstance(parsed, list):
                        new_text = "# 🎮 PROPUESTA DE DISEÑO ESTRATÉGICO\n\n"
                        for item in parsed:
                            if isinstance(item, dict):
                                # Try to find a title/heading in multiple languages/keys
                                head = item.get('recommendation', item.get('title', item.get('titulo', item.get('heading', item.get('recomendacion', '')))))
                                body = item.get('description', item.get('content', item.get('descripcion', item.get('contenido', item.get('body', '')))))
                                
                                # If still empty, use the first available values
                                if not head and item.values():
                                    head = list(item.values())[0] if not head else head
                                    if len(item.values()) > 1 and not body:
                                        body = list(item.values())[1]
                                
                                new_text += f"### {head}\n{body}\n\n"
                            else:
                                new_text += f"{item}\n\n"
                    # Case B: Object with sections (Gemini style)
                    elif isinstance(parsed, dict):
                        title = parsed.get('title', parsed.get('propuesta', parsed.get('titulo', 'PROPUESTA DE JUEGO')))
                        new_text = f"# 🎮 {title}\n\n"
                        
                        # Handle "sections" key
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
                            # Direct key-value display for flat objects
                            for k, v in parsed.items():
                                if k.lower() not in ['title', 'sections', 'propuesta', 'titulo', 'secciones']:
                                    new_text += f"### {k.replace('_', ' ').upper()}\n{v}\n\n"
                    
                    if new_text:
                        text = new_text
                except:
                    pass
            
            clean_proposals[model_name] = text
            
        return {
            "winners": winners,
            "proposals": clean_proposals,
            "research_context": self.current_research_context
        }
