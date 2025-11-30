from ast import Tuple
import sys
sys.path.append('..')
from kg_connector.kg_connector import KGConnector
from utils.sim import Similarity
from typing import Optional
import dotenv
import os
import numpy as np
dotenv.load_dotenv()

class RelationEmbedder:
    def __init__(self, kg_connector: Optional[KGConnector] = None, sim: Optional[Similarity] = None, store_path: str = os.getenv('EMBEDDING_STORAGE_PATH', '')) -> None:

        if kg_connector is None:
            self.kg_connector = KGConnector()
        else:
            self.kg_connector = kg_connector

        if sim is None:
            self.sim = Similarity()
        else:
            self.sim = sim

        if not store_path:
            raise ValueError("EMBEDDING_STORAGE_PATH environment variable is not set.")
        else:
            self.store_path = store_path

        # Check if the store_path directory exists, if not create it
        if not os.path.exists(self.store_path):
            print(f"Creating directory for embeddings at {self.store_path}")
            os.makedirs(self.store_path)

        # Check if the 2 file of embeddings exists in the store_path, if not generate them
        self.npy_file = os.path.join(self.store_path, os.getenv('EMBEDDING_FILENAME', 'kg_relations') + '.npy')
        self.txt_file = os.path.join(self.store_path, os.getenv('EMBEDDING_FILENAME', 'kg_relations') + '.txt')
        if not os.path.isfile(self.npy_file) or not os.path.isfile(self.txt_file):
            print(f"Embeddings files not found in {self.store_path}. Generating new embeddings.")
            self.embed_relations(verbose=True)

        self.kg_relations = [line.strip() for line in open(self.txt_file, 'r', encoding='utf-8')]
        self.relation_embeddings = np.load(self.npy_file)
        print(f"Loaded {len(self.kg_relations)} relations and their embeddings from {self.store_path}.")

    def embed_relations(self, save_to: Optional[str] = None, file_name: str = os.getenv('EMBEDDING_FILENAME', 'kg_relations'), verbose = False) -> tuple:
        if save_to is None:
            save_to = self.store_path
        relations = self.kg_connector.load_kg_relations()
        if verbose:
            print(f"Loaded {len(relations)} relations from KG.")
        save_to = os.path.join(save_to, file_name)
        embedings = self.sim.embed(relations, save_to=save_to)
        if verbose:
            print(f"Generated embedings for {len(embedings)} relations.")
        return relations, embedings

    def match_relation(self, raw_relation: str, embedding_storage: Optional[str] = None, top_k: int = 1):
        if embedding_storage is None:
            embedding_storage = self.store_path
        return self.sim.match_embed(raw_relation, self.kg_relations, self.relation_embeddings, top_k=top_k)


