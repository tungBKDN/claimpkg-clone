import sys
sys.path.append("..")
from typing import Dict, List, Optional, Tuple

from kg_connector.kg_connector import KGConnector
from utils.sim import Similarity

class Greedy:
    def __init__(self, kg_connector: KGConnector, sim: Optional[Similarity] = None):
        self.kg_connector = kg_connector
        self.sim = sim if sim is not None else Similarity()

    def greedy_2(self, standardized_triplets: list[dict[str, str]], greedy_level: int = 1) -> str:
        """
        Return True if any triplet supports the claim.
        """
        explicit_entities = []
        for triplet in standardized_triplets:
            if not triplet["head"].startswith("unknown_"):
                explicit_entities.append(triplet["head"])
            if not triplet["tail"].startswith("unknown_"):
                explicit_entities.append(triplet["tail"])
        explicit_entities = list(set(explicit_entities))

        related_entities = self.kg_connector.get_children_up_to_k(entities=explicit_entities, k=greedy_level)
        # Get all relations between related entities
        greedy_triplets = self.kg_connector.get_relations_between(entities=list(related_entities))
        return greedy_triplets

    def greedy(self, triplets: list[dict[str, str]], k_relations = 4) -> str:
        """
        Params:
            triplets: list of non-standardized triplet strings
            k_relations: number of relations to retrieve per entity
        """

        parsed = []
        completed = set()

        for triplet in triplets:
            head = triplet["head"]
            relation = triplet["relation"]
            tail = triplet["tail"]
            parsed.append((head, relation, tail))

        # entity_to_entity : List[Tuple[str, str]]= [] # Store all direct entity to entity relations
        # for triplet in parsed:
        #     if not triplet[0].startswith("unknown_") and not triplet[2].startswith("unknown_"):
        #         entity_to_entity.append( (triplet[0], triplet[2]) )

        # These triplets are not completed yet, so we need to complete them
        entity_to_rel : Dict[str, List[str]] = dict()
        # Eg: {ent_1: [rel_1, rel_2], ...}
        for triplet in parsed:
            if not triplet[0].startswith("unknown_"):
                if triplet[0] not in entity_to_rel:
                    entity_to_rel[triplet[0]] = []
                entity_to_rel[triplet[0]].append(triplet[1])
            if not triplet[2].startswith("unknown_"):
                if triplet[2] not in entity_to_rel:
                    entity_to_rel[triplet[2]] = []
                entity_to_rel[triplet[2]].append(triplet[1])

        # Change relations to candidates of relations
        relations = set()
        for rel_list in entity_to_rel.values():
            for rel in rel_list:
                relations.add(rel)
        relations = list(relations)
        relation_candidates : Dict[str, List[str]] = dict()
        for rel in relations:
            relation_candidates[rel] = self.sim.get_candidate_relations(raw_relation=rel, top_k=k_relations)
        # Map back to entity_to_rel
        for entity, rel_list in entity_to_rel.items():
            candidate_list = []
            for rel in rel_list:
                candidate_list.extend([x[0] for x in relation_candidates[rel]])
            entity_to_rel[entity] = list(set(candidate_list))

        neighbors : Dict[str, Dict[str, List[str]]] = dict()
        # Dict[entity, Dict[relation, List[neighbor_entities]]]
        for entity, rel_list in entity_to_rel.items():
            neighbor_dict = self.kg_connector.get_neighbors_by_relations(entity=entity, relations=rel_list)
            neighbors[entity] = neighbor_dict

        # Now, build the greedy triplets
        count = 0
        greedy_triplets = []
        for entity, rel_dict in neighbors.items():
            for relation, neighbor_entities in rel_dict.items():
                for neighbor in neighbor_entities:
                    triplet_str = f"<e>{entity}</e> || {relation} || <e>{neighbor}</e>"
                    if triplet_str not in completed:
                        greedy_triplets.append(triplet_str)
                        completed.add(triplet_str)
                        count += 1

        return "\n".join(greedy_triplets)