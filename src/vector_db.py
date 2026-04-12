import chromadb
import hashlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from .config import CHROMA_DB_PATH, VERBOSE

# Singleton for encoder and client
_encoder = None
_chroma_client = None
_query_embedding_cache = {}
_item_embedding_cache = {}

def get_encoder():
    global _encoder
    if _encoder is None:
        if VERBOSE: print("Cargando encoder (all-mpnet-base-v2)...")
        _encoder = SentenceTransformer('all-mpnet-base-v2')
    return _encoder

def get_chromadb_client():
    global _chroma_client
    if _chroma_client is None:
        if not CHROMA_DB_PATH.exists():
            CHROMA_DB_PATH.mkdir(parents=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return _chroma_client

def get_query_embedding_cached(query):
    global _query_embedding_cache
    query_hash = hashlib.md5(query.encode()).hexdigest()
    if query_hash not in _query_embedding_cache:
        encoder = get_encoder()
        _query_embedding_cache[query_hash] = encoder.encode([query])[0]
        if len(_query_embedding_cache) > 1000:
            _query_embedding_cache.pop(next(iter(_query_embedding_cache)))
    return _query_embedding_cache[query_hash]

def get_item_embedding_cached(title, description):
    global _item_embedding_cache
    cache_key = f"{title}|{description}"
    key_hash = hashlib.md5(cache_key.encode()).hexdigest()
    if key_hash not in _item_embedding_cache:
        encoder = get_encoder()
        item_text = f"{title}. {description}"
        _item_embedding_cache[key_hash] = encoder.encode([item_text])[0]
        if len(_item_embedding_cache) > 5000:
            _item_embedding_cache.pop(next(iter(_item_embedding_cache)))
    return _item_embedding_cache[key_hash]

def initialize_vector_db(df, dataset_name='serious_games', force_recreate=False):
    client = get_chromadb_client()
    collection_name = f"{dataset_name}_collection"
    
    if force_recreate:
        try:
            client.delete_collection(name=collection_name)
        except: pass

    try:
        collection = client.get_collection(name=collection_name)
        if collection.count() > 0:
            if VERBOSE: print(f"✅ Colección '{collection_name}' cargada con {collection.count()} items.")
            return collection
    except:
        if VERBOSE: print(f"📦 Creando nueva colección '{collection_name}'...")
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:search_ef": 100, "hnsw:M": 32, "hnsw:space": "cosine"}
        )

    encoder = get_encoder()
    embeddings = encoder.encode(df['full_text'].tolist(), show_progress_bar=True)
    
    batch_size = 5000
    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        batch_df = df.iloc[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]
        
        ids = [f"{dataset_name}_{idx}" for idx in batch_df.index]
        documents = batch_df['title'].fillna('').astype(str).tolist()
        metadatas = [
            {'id': str(row['item_id']), 'description': str(row['description']), 'item_type': str(row['item_type'])}
            for _, row in batch_df.iterrows()
        ]
        
        collection.add(embeddings=batch_embeddings.tolist(), documents=documents, metadatas=metadatas, ids=ids)
    
    return collection

def rag_retrieval(query, collection, top_k=30, where=None):
    query_emb = get_query_embedding_cached(query)
    results = collection.query(
        query_embeddings=[query_emb.tolist()], 
        n_results=top_k, 
        include=['documents', 'metadatas', 'distances'], 
        where=where
    )

    if not results['ids'] or len(results['ids'][0]) == 0:
        return pd.DataFrame(columns=['id', 'title', 'description', 'item_type', 'similarity'])

    return pd.DataFrame({
        'id': [m['id'] for m in results['metadatas'][0]],
        'title': results['documents'][0],
        'description': [m['description'] for m in results['metadatas'][0]],
        'item_type': [m['item_type'] for m in results['metadatas'][0]],
        'similarity': [1 - d for d in results['distances'][0]]
    })
