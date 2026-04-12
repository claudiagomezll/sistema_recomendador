import pandas as pd
import os
from .config import DATASET_PATH, VERBOSE

def load_data_gamification():
    """Carga los datasets específicos de gamificación y serious games."""
    read_params = {'sep': None, 'engine': 'python', 'on_bad_lines': 'warn'}
    
    try:
        dynamics = pd.read_csv(DATASET_PATH / "dynamicsGam.csv", **read_params)
        type_dynamics = pd.read_csv(DATASET_PATH / "typeDynamicsGam.csv", **read_params)
        elements = pd.read_csv(DATASET_PATH / "elementsGam.csv", **read_params)
        mechanics = pd.read_csv(DATASET_PATH / "mechanicsGam.csv", **read_params)
        narratives = pd.read_csv(DATASET_PATH / "narrativesGam.csv", **read_params)

        dynamics = dynamics.merge(type_dynamics, on="typeId", how="left")
        
        dynamics_df = pd.DataFrame({
            "item_id": dynamics["dynamicId"].astype(str),
            "item_type": "dynamic",
            "title": dynamics["dynamicGam"],
            "description": dynamics.apply(
                lambda r: f"Dynamic: {r['dynamicGam']}. Type: {r.get('dynamicType', '')}", axis=1
            ),
            "type_id": dynamics["typeId"],
            "type_label": dynamics["dynamicType"]
        })

        elements_df = pd.DataFrame({
            "item_id": elements["elementId"].astype(str),
            "item_type": "element",
            "title": elements["elementGam"],
            "description": elements["elementGam"],
            "type_id": None,
            "type_label": None
        })

        mechanics_df = pd.DataFrame({
            "item_id": mechanics["mechanicId"].astype(str),
            "item_type": "mechanic",
            "title": mechanics["mechanicGam"],
            "description": mechanics["mechanicGam"],
            "type_id": None,
            "type_label": None
        })

        narratives_df = pd.DataFrame({
            "item_id": narratives["narrativeId"].astype(str),
            "item_type": "narrative",
            "title": narratives["narrativeGamTitle"],
            "description": narratives["narrativeGamTitle"],
            "type_id": None,
            "type_label": None
        })

        df = pd.concat([dynamics_df, elements_df, mechanics_df, narratives_df], ignore_index=True)
        df["item_id"] = df["item_id"].astype(str)
        df["global_item_id"] = df["item_type"] + "_" + df["item_id"]
        
        df["full_text"] = (
            "Type: " + df["item_type"].fillna("") +
            ". Title: " + df["title"].fillna("") +
            ". Description: " + df["description"].fillna("") +
            ". Category: " + df["type_label"].fillna("")
        )

        if VERBOSE:
            print(f"✅ Dataset cargado: {len(df)} items totales.")
        
        return df
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")
        return None

def load_serious_games_ratings():
    """Carga los ratings de intervención para filtrado colaborativo."""
    try:
        elements_i = pd.read_csv(DATASET_PATH / "elementsIntervention.csv").rename(columns=lambda x: x.strip())
        dynamics_i = pd.read_csv(DATASET_PATH / "dynamicsIntervention.csv").rename(columns=lambda x: x.strip())
        mechanics_i = pd.read_csv(DATASET_PATH / "mechanicsIntervention.csv").rename(columns=lambda x: x.strip())
        narratives_i = pd.read_csv(DATASET_PATH / "narrativesIntervention.csv").rename(columns=lambda x: x.strip())

        elements_i["item_type"] = "element"
        elements_i = elements_i.rename(columns={"elementId": "item_id"})
        
        dynamics_i["item_type"] = "dynamic"
        dynamics_i = dynamics_i.rename(columns={"dynamicId": "item_id"})
        
        mechanics_i["item_type"] = "mechanic"
        mechanics_i = mechanics_i.rename(columns={"mechanicId": "item_id"})
        
        narratives_i["item_type"] = "narrative"
        narratives_i = narratives_i.rename(columns={"narrativeId": "item_id"})

        ratings_df = pd.concat([elements_i, dynamics_i, mechanics_i, narratives_i], ignore_index=True)
        ratings_df["item_id"] = ratings_df["item_id"].astype(str).str.strip()
        ratings_df["item_type"] = ratings_df["item_type"].astype(str).str.strip().str.lower()
        ratings_df["global_item_id"] = ratings_df["item_type"] + "_" + ratings_df["item_id"]

        if VERBOSE:
            print(f"✅ Ratings cargados: {len(ratings_df)} registros.")
            
        return ratings_df
    except Exception as e:
        print(f"⚠️ Usando ratings sintéticos por error: {e}")
        return pd.DataFrame({
            'userId': [1]*5, 
            'item_id': [1,2,3,4,5], 
            'item_type': ['dynamic']*5, 
            'rating': [5,4,5,4,5], 
            'global_item_id': ['dynamic_1','dynamic_2','dynamic_3','dynamic_4','dynamic_5']
        })
