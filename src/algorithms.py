import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from .config import USE_RAPIDFUZZ, USE_VECTORIZED_MMR, VERBOSE, DATASETS
from .vector_db import get_encoder

# Try to import rapidfuzz if available
if USE_RAPIDFUZZ:
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        USE_RAPIDFUZZ = False

# --- Collaborative Filtering (FCD) ---

def load_user_ratings_matrix(ratings_df, item_col='global_item_id'):
    unique_users = sorted(ratings_df['userId'].unique())
    unique_items = sorted(ratings_df[item_col].unique())

    user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_id_map = {item_id: idx for idx, item_id in enumerate(unique_items)}

    row_indices = [user_id_map[uid] for uid in ratings_df['userId']]
    col_indices = [item_id_map[iid] for iid in ratings_df[item_col]]
    ratings_values = ratings_df['rating'].values

    user_item_matrix = csr_matrix(
        (ratings_values, (row_indices, col_indices)),
        shape=(len(unique_users), len(unique_items))
    )
    return user_item_matrix, user_id_map, item_id_map

def fcd_recommendations(user_id, ratings_df, user_item_matrix, user_id_map, item_id_map, top_k=30):
    if user_id not in user_id_map:
        return []

    user_idx = user_id_map[user_id]
    user_vector = user_item_matrix[user_idx]
    
    # Calculate similarities
    similarities = cosine_similarity(user_vector, user_item_matrix).flatten()
    similarities[user_idx] = -1
    top_indices = np.argsort(similarities)[-50:][::-1]
    similar_users = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]

    if not similar_users:
        return []

    consumed_items = set(ratings_df[ratings_df['userId'] == user_id]['global_item_id'].values)
    item_scores = {}

    for similar_user_idx, similarity in similar_users:
        similar_user_items = user_item_matrix[similar_user_idx].nonzero()[1]
        for item_idx in similar_user_items:
            item_id = [iid for iid, idx in item_id_map.items() if idx == item_idx][0]
            if item_id in consumed_items:
                continue
            rating = user_item_matrix[similar_user_idx, item_idx]
            if rating >= 4.0:
                if item_id not in item_scores:
                    item_scores[item_id] = {'total_score': 0, 'total_sim': 0}
                item_scores[item_id]['total_score'] += similarity * rating
                item_scores[item_id]['total_sim'] += similarity

    recommendations = []
    for item_id, scores in item_scores.items():
        if scores['total_sim'] > 0:
            fcd_score = scores['total_score'] / scores['total_sim']
            fcd_score_norm = (fcd_score - 1) / 4
            recommendations.append({
                'item_id': str(item_id),
                'fcd_score': fcd_score_norm,
                'source': 'fcd',
                'n_recommendations': int(scores['total_sim'])
            })
    
    recommendations.sort(key=lambda x: x['fcd_score'], reverse=True)
    return recommendations[:top_k]

# --- Fuzzy Matching ---

def fuzzy_match_title(title, df, threshold=0.85):
    if USE_RAPIDFUZZ:
        result = process.extractOne(
            title, df['title'].tolist(), scorer=fuzz.ratio, score_cutoff=threshold * 100
        )
        if result:
            matched_title, score_100, idx = result
            return matched_title, score_100 / 100, df.iloc[idx]
    else:
        from difflib import SequenceMatcher
        best_score = 0
        best_match = None
        best_item = None
        for _, row in df.iterrows():
            score = SequenceMatcher(None, title, row['title']).ratio()
            if score > best_score:
                best_score = score
                best_match = row['title']
                best_item = row
        if best_score >= threshold:
            return best_match, best_score, best_item
    return None, 0, None

# --- Ranking ---

def calculate_weights(user_id=None, ratings_df=None):
    if user_id is None or ratings_df is None:
        return {'llm': 1.0, 'fcd': 0.0}
    
    n_ratings = len(ratings_df[ratings_df['userId'] == user_id])
    if n_ratings >= 50:
        return {'llm': 0.6, 'fcd': 0.4}
    elif n_ratings >= 10:
        return {'llm': 0.7, 'fcd': 0.3}
    else:
        return {'llm': 0.9, 'fcd': 0.1}

def calculate_ranking_rrf(validated_items, weights=None):
    if weights is None: weights = {'llm': 1.0, 'fcd': 0.0}
    if not validated_items: return []
    
    # Simple weighted RRF-like logic
    item_groups = {}
    for item in validated_items:
        title = item['title']
        if title not in item_groups:
            item_groups[title] = {'relevance': 0, 'consensus': 0, 'fcd': 0, 'data': item}
        
        item_groups[title]['relevance'] = max(item_groups[title]['relevance'], item.get('relevance_score', 0))
        if item['llm_source'] != 'fcd':
            item_groups[title]['consensus'] += 1
        else:
            item_groups[title]['fcd'] = max(item_groups[title]['fcd'], item.get('fcd_score', 0))

    max_mentions = max([g['consensus'] for g in item_groups.values()]) or 1
    ranked = []
    for title, scores in item_groups.items():
        rel = scores['relevance']
        cons = scores['consensus'] / max_mentions
        fcd = scores['fcd']
        
        # Formula: combine relevance, consensus and fcd
        # Final score calculation and casting to float for JSON compatibility
        final_score = float((rel * weights['llm']) + (fcd * weights['fcd']) + (cons * 0.1))
        
        # Update rank_score and return only essential data
        item_data = scores['data']
        item_data['rank_score'] = final_score
        ranked.append(item_data)
        
    ranked.sort(key=lambda x: x['rank_score'], reverse=True)
    return ranked

# --- MMR Diversification ---

def mmr_diversification(ranked_items, k=10, lambda_param=0.7):
    if len(ranked_items) <= k: return ranked_items
    
    encoder = get_encoder()
    titles = [item['title'] for item in ranked_items]
    # We should use pre-calculated embeddings or calculate them here
    # For simplicity, calculate them here if not provided
    embeddings_list = [encoder.encode([f"{item['title']}. {item['description']}"])[0] for item in ranked_items]
    embeddings_matrix = np.array(embeddings_list)
    
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    embeddings_norm = embeddings_matrix / norms
    
    relevances = np.array([item['rank_score'] for item in ranked_items])
    selected_indices = [0]
    remaining_indices = list(range(1, len(ranked_items)))
    
    while len(selected_indices) < k and remaining_indices:
        selected_embs = embeddings_norm[selected_indices]
        remaining_embs = embeddings_norm[remaining_indices]
        
        similarities = remaining_embs @ selected_embs.T
        max_sims = similarities.max(axis=1)
        
        mmr_scores = lambda_param * relevances[remaining_indices] + (1 - lambda_param) * (1 - max_sims)
        best_idx = mmr_scores.argmax()
        selected_indices.append(remaining_indices.pop(best_idx))
    
    return [ranked_items[i] for i in selected_indices]
