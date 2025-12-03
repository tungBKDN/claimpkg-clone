
import sys
sys.path.append('..')
from utils.sim import Similarity
from middle.group_n_decompose import GroupNDecompose
from typing import List, Dict, Optional, Callable

from kg_connector.kg_connector import KGConnector

class RetrieveAndUnion:
    def __init__(self, kg_connector: Optional[KGConnector] = None):
        self.kg_connector = kg_connector if kg_connector else KGConnector()
        self.sim = Similarity()

    def retrieve_neighbors(self, entity_names: list[str]):
        all_neighbors = []
        retrived_neighbors = self.kg_connector.get_neighbors(entity_names)
        return retrived_neighbors

    def flatten_neighbors(self, neighbor_dict: Dict) -> List:
        """
        neighbor_dict: dict {relation: [entities]}
        → return flat candidate entity list
        """
        candidates = set()
        for rel, vals in neighbor_dict.items():
            for v in vals:
                candidates.add(v)
        return list(candidates)

    def build_candidates(self, neighbor_info: Dict) -> Dict:
        flatten_neighbors = {}
        keys = neighbor_info.keys()
        for k in keys:
            flatten_neighbors[k] = self.flatten_neighbors(neighbor_info[k])
        return flatten_neighbors

    def resolve_unknown_with_relation(
        self,
        neighbors: Dict[str, Dict[str, List[str]]],
        pseudo_graph: Dict
    ) -> List[Dict]:
        """
        ClaimPKG-compliant subgraph retrieval for unknowns WITH real relations.
        Returns only: triplets, candidate_nodes_after_top_k, explicit_nodes
        """

        incomplete_groups = pseudo_graph.get("incomplete_groups", {})
        result = []

        for unk_id, group in incomplete_groups.items():

            explicit_entities = group.get("explicit_entities", [])
            pseudo_relations  = group.get("relations", [])

            # Skip null-rel groups (paper handles separately)
            if all(r.startswith("null_relation") for r in pseudo_relations):
                continue

            # -------------------------
            # Build candidate sets Cu
            # -------------------------
            candidate_sets = []
            for e in explicit_entities:
                if e not in neighbors:
                    candidate_sets.append([])
                    continue

                cand_list = []
                for rel, vals in neighbors[e].items():
                    for v in vals:
                        if isinstance(v, str) and "," in v and '"' in v:
                            continue
                        cand_list.append(v)

                candidate_sets.append(cand_list)

            # -------------------------
            # Build KG for ranker
            # -------------------------
            KG = {
                ent: [(rel, nb) for rel, lst in rdict.items() for nb in lst]
                for ent, rdict in neighbors.items()
            }

            # -------------------------
            # Rank candidates (Eq.(5-6))
            # -------------------------
            ranked = self.sim.rank_candidates(
                candidate_sets=candidate_sets,
                explicit_entities=explicit_entities,
                pseudo_relations=pseudo_relations,
                KG=KG,
                sim_func=self.sim.sim,
                k1=3,
                normalize=True,
                aggregate="max"
            )

            top_nodes = [c for c, _ in ranked]  # top-K candidates

            # -------------------------
            # Build union graph S* (pure KG)
            # -------------------------
            S = []

            for e in explicit_entities:
                if e not in neighbors:
                    continue

                for rel, vals in neighbors[e].items():
                    for v in vals:
                        if v in top_nodes:
                            S.append(f"<e>{e}</e>||{rel}||<e>{v}</e>")

            # Deduplicate triplets
            S = list(set(S))

            # -------------------------------------------------
            # Step 4 — Append per-unknown result
            # -------------------------------------------------
            result.append({
                "triplets": S,
                "candidate_nodes_after_top_k": top_nodes,
                "explicit_nodes": explicit_entities
            })
        return result

    def resolve_unknown_without_relation(
        self,
        neighbors: Dict[str, Dict[str, List[str]]],
        pseudo_graph: Dict,
        k1: int = 3
    ) -> List[Dict]:
        """
        Resolve unknown entities whose relations are null_relation_X.
        According to ClaimPKG, null-rel groups use frequency-based scoring
        instead of relation-aware ranking.

        Returns a list of dicts:
        [
            {
                "triplets": [...],
                "candidate_nodes_after_top_k": [...],
                "explicit_nodes": [...]
            },
            ...
        ]
        """

        incomplete_groups = pseudo_graph.get("incomplete_groups", {})
        results = []

        for unk_id, group in incomplete_groups.items():

            explicit_entities = group.get("explicit_entities", [])
            pseudo_relations  = group.get("relations", [])

            # Only handle groups where ALL relations are null_relation_x
            if not all(r.startswith("null_relation") for r in pseudo_relations):
                continue

            # -------------------------------------------------
            # Step 1 — Collect candidate sets (neighbors only)
            # -------------------------------------------------
            candidate_list = []   # flatten all candidates
            candidate_sets = []   # keep per-entity lists

            for e in explicit_entities:
                if e not in neighbors:
                    candidate_sets.append([])
                    continue

                cand = []
                for rel, vals in neighbors[e].items():
                    for v in vals:
                        if isinstance(v, str) and "," in v and '"' in v:
                            continue
                        cand.append(v)

                candidate_sets.append(cand)
                candidate_list.extend(cand)

            # -------------------------------------------------
            # Step 2 — Frequency-based scoring (ClaimPKG null-rel)
            # -------------------------------------------------
            from collections import Counter

            freq = Counter(candidate_list)
            top_candidates = [c for c, _ in freq.most_common(k1)]

            # -------------------------------------------------
            # Step 3 — Build union subgraph S* (KG-only)
            # -------------------------------------------------
            S = []

            for e in explicit_entities:
                if e not in neighbors:
                    continue

                for rel, vals in neighbors[e].items():
                    for v in vals:
                        if v in top_candidates:
                            S.append(f"<e>{e}</e>||{rel}||<e>{v}</e>")

            # Deduplicate triplets
            S = list(set(S))

            # -------------------------------------------------
            # Step 4 — Append per-unknown result
            # -------------------------------------------------
            results.append({
                "triplets": S,
                "candidate_nodes_after_top_k": top_candidates,
                "explicit_nodes": explicit_entities
            })

        return results

    def retrive_and_union(self, standardized_triplets: List[str], group_n_decomposed: Dict) -> List:
        entities = set()
        from utils.parser import str_to_triplet, union_triplets
        for triplet in standardized_triplets:
            head, rel, tail = str_to_triplet(triplet)
            if not head.startswith("unknown_"):
                entities.add(head)
            if not tail.startswith("unknown_"):
                entities.add(tail)

        neighbors = self.retrieve_neighbors(list(entities))
        candidates = self.build_candidates(neighbors)
        filled_unk_with_relations = self.resolve_unknown_with_relation(
            neighbors=neighbors,
            pseudo_graph=group_n_decomposed,
        )

        filled_unk_without_relations = self.resolve_unknown_without_relation(
            neighbors=neighbors,
            pseudo_graph=group_n_decomposed,
        )

        unified_results = union_triplets(filled_with_rel=filled_unk_with_relations, filled_without_rel=filled_unk_without_relations)

        return unified_results

