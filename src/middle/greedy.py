import sys
sys.path.append("..")

from kg_connector.kg_connector import KGConnector

class Greedy:
    def __init__(self, kg_connector: KGConnector):
        self.kg_connector = kg_connector

    def greedy(self, standardized_triplets: list[dict[str, str]], greedy_level: int = 2) -> str:
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

