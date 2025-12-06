import sys
sys.path.append('..')
from typing import List, Dict, Optional, Callable, Tuple, Any
from collections import defaultdict, Counter
import heapq

from kg_connector.kg_connector import KGConnector
from utils.sim import Similarity


class RetrieveAndUnion:
    """
    Full implementation of the ClaimPKG subgraph retrieval / union logic
    - Implements Eq.(5) and Eq.(6) style scoring and per-set top-k selection.
    - Handles "literal" entity names (e.g., '"Affluence"') as valid entity keys.
    - Builds reverse neighbor mapping to support 2-way lookup (to avoid missing candidates).
    - Uses self.sim.score(...) when available (preferred), else uses sim.sim pairwise.
    - Keeps all entities intact (does NOT discard entities containing commas or quotes).
    - Merges per-group results into one consolidated output and force-includes KG facts
      for all explicit entities to avoid losing evidence.
    """

    def __init__(self, kg_connector: Optional[KGConnector] = None):
        self.kg_connector = kg_connector if kg_connector else KGConnector()
        self.sim = Similarity()

    # -------------------------
    # Utilities: neighbor retrieval & local KG
    # -------------------------
    def retrieve_neighbors(self, entity_names: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Retrieve neighbors for a list of entity names from KGConnector.
        Returns the forward neighbor map only (reverse map can be built separately).
        """
        forward = self.kg_connector.get_neighbors(entity_names)
        for e in entity_names:
            if e not in forward:
                forward[e] = {}
        return forward

    def _build_reverse_neighbors(self, neighbors: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Build reverse mapping: for each triple (src -rel-> tgt) in neighbors,
        add reverse[tgt][rel].append(src).
        """
        rev = defaultdict(dict)
        for src, relmap in neighbors.items():
            for rel, vals in relmap.items():
                for v in vals:
                    if rel not in rev[v]:
                        rev[v][rel] = []
                    rev[v][rel].append(src)
        return dict(rev)

    def _build_local_KG(self, neighbors: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        { head_entity: [(relation, tail_entity), ...], ... }
        """
        KG = {}
        for ent, relmap in neighbors.items():
            items: List[Tuple[str, str]] = []
            for rel, vals in relmap.items():
                for v in vals:
                    items.append((rel, v))
            KG[ent] = items
        return KG

    # -------------------------
    # Parsing triplet strings
    # -------------------------
    def _parse_triplet_str(self, s: str) -> Tuple[str, str, str]:
        parts = s.split("||")
        if len(parts) != 3:
            raise ValueError(f"Bad triplet format: {s}")
        def strip_entity(x: str) -> str:
            t = x.strip()
            if t.startswith("<e>") and t.endswith("</e>"):
                return t[3:-4]
            return t
        return strip_entity(parts[0]), parts[1].strip(), strip_entity(parts[2])

    # -------------------------
    # Candidate scoring (Eq.5 style)
    # -------------------------
    def _score_candidate_using_simscore(
        self,
        candidate: str,
        explicit_entities: List[str],
        pseudo_relations: List[str],
        KG: Dict[str, List[Tuple[str, str]]],
        normalize: bool = False
    ) -> float:
        """
        Use self.sim.score(...) if available (preferred). Otherwise fallback to pairwise simulation.
        normalize param forwarded to sim.score if available; fallback will normalize or not accordingly.
        """
        if hasattr(self.sim, "score"):
            try:
                return float(self.sim.score(
                    candidate_entity=candidate,
                    explicit_entities=explicit_entities,
                    pseudo_relations=pseudo_relations,
                    KG=KG,
                    sim_func=self.sim.sim,
                    normalize=normalize
                ))
            except Exception:
                pass

        # fallback
        total = 0.0
        matches = 0
        for e_ui, r_ui in zip(explicit_entities, pseudo_relations):
            for r_act, nb in KG.get(e_ui, []):
                if nb == candidate:
                    try:
                        total += float(self.sim.sim(r_ui, r_act))
                    except Exception:
                        total += 0.0
                    matches += 1
        if normalize and matches > 0:
            return total / matches
        return total

    # -------------------------
    # _retrieve_complete_triplets (k2)
    # -------------------------
    def _retrieve_complete_triplets(
        self,
        neighbors: Dict[str, Dict[str, List[str]]],
        pseudo_graph: Dict[str, Any],
        k2: int = 1
    ) -> List[str]:
        results: List[str] = []
        KG = self._build_local_KG(neighbors)

        triples: List[Tuple[str, str, str]] = []
        if "complete_triplets" in pseudo_graph and pseudo_graph["complete_triplets"]:
            for t in pseudo_graph["complete_triplets"]:
                if isinstance(t, str):
                    try:
                        triples.append(self._parse_triplet_str(t))
                    except Exception:
                        continue
                elif isinstance(t, (list, tuple)) and len(t) == 3:
                    triples.append((t[0], t[1], t[2]))
        else:
            for t in pseudo_graph.get("triplets", []):
                try:
                    head, rel, tail = self._parse_triplet_str(t)
                    if not head.startswith("unknown_") and not tail.startswith("unknown_"):
                        triples.append((head, rel, tail))
                except Exception:
                    continue

        for head, rel_pseudo, tail in triples:
            rels_found: List[Tuple[float, str]] = []
            for rel_actual, vals in neighbors.get(head, {}).items():
                if tail in vals:
                    try:
                        score = float(self.sim.sim(rel_pseudo, rel_actual))
                    except Exception:
                        score = 0.0
                    rels_found.append((score, rel_actual))
            if not rels_found:
                continue
            rels_found.sort(key=lambda x: x[0], reverse=True)
            for score, rel_actual in rels_found[:k2]:
                results.append(f"<e>{head}</e>||{rel_actual}||<e>{tail}</e>")

        # dedup preserve order
        seen = set()
        dedup = []
        for r in results:
            if r not in seen:
                dedup.append(r); seen.add(r)
        return dedup

    # -------------------------
    # Core: resolve unknowns that HAVE real relations (not null_relation)
    # -------------------------
    def resolve_unknown_with_relation(
        self,
        neighbors: Dict[str, Dict[str, List[str]]],
        pseudo_graph: Dict,
        k1: int = 5,
        k2: int = 1,
        normalize_scores: bool = False,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Resolve unknown groups with relations. Returns merged single-item list:
        [{
            "triplets": [...],
            "candidate_nodes_after_top_k": [...],
            "explicit_nodes": [...]
        }]
        Merges per-group results to avoid losing groups.
        """
        incomplete_groups = pseudo_graph.get("incomplete_groups", {})
        per_group_results = []
        KG = self._build_local_KG(neighbors)
        reverse_neighbors = self._build_reverse_neighbors(neighbors)

        for unk_id, group in incomplete_groups.items():
            explicit_entities: List[str] = group.get("explicit_entities", [])
            pseudo_relations: List[str] = group.get("relations", [])

            if all(r.startswith("null_relation") for r in pseudo_relations):
                if verbose:
                    print(f"[resolve] skip {unk_id} (all null_relation)")
                continue

            if verbose:
                print(f"\n[resolve] Processing {unk_id}")
                print(" explicit_entities:", explicit_entities)
                print(" pseudo_relations:", pseudo_relations)

            # build per-entity candidate pairs (forward + reverse)
            per_entity_candidates: List[List[Tuple[str, str, str]]] = []
            for e in explicit_entities:
                cand_pairs: List[Tuple[str, str, str]] = []
                if e in neighbors:
                    for rel_actual, vals in neighbors[e].items():
                        for v in vals:
                            cand_pairs.append((v, rel_actual, "forward"))
                if e in reverse_neighbors:
                    for rel_actual, sources in reverse_neighbors[e].items():
                        for src in sources:
                            cand_pairs.append((src, rel_actual, "reverse"))
                per_entity_candidates.append(cand_pairs)

            if verbose:
                for i, e in enumerate(explicit_entities):
                    print(f"  candidates for [{i}] '{e}': {per_entity_candidates[i]}")

            if all(len(c) == 0 for c in per_entity_candidates):
                if verbose:
                    print("  -> no candidates for any explicit in this group")
                per_group_results.append({
                    "triplets": [],
                    "candidate_nodes_after_top_k": [],
                    "explicit_nodes": explicit_entities
                })
                continue

            # collect all candidate entities
            all_candidates = set()
            for lst in per_entity_candidates:
                for cand_entity, _, _ in lst:
                    all_candidates.add(cand_entity)

            if verbose:
                print("  all unique candidates:", all_candidates)

            # compute candidate scores
            candidate_scores: Dict[str, float] = {}
            for cand in all_candidates:
                candidate_scores[cand] = self._score_candidate_using_simscore(
                    candidate=cand,
                    explicit_entities=explicit_entities,
                    pseudo_relations=pseudo_relations,
                    KG=KG,
                    normalize=normalize_scores
                )

            if verbose:
                print("  candidate_scores:")
                for c, sc in sorted(candidate_scores.items(), key=lambda x: -x[1]):
                    print(f"    {c}: {sc:.6f}")

            # per-explicit top-k1 selection using global candidate_scores
            selected_candidates_ordered: List[str] = []
            selected_candidates_set = set()
            selected_triplets: List[str] = []

            for idx, e in enumerate(explicit_entities):
                cand_pairs = per_entity_candidates[idx]
                if not cand_pairs:
                    if verbose:
                        print(f"  explicit '{e}' no candidates")
                    continue

                # keep best rel/direction for each cand_entity
                per_map: Dict[str, Tuple[float, str, str]] = {}
                for cand_entity, rel_actual, direction in cand_pairs:
                    sc = candidate_scores.get(cand_entity, 0.0)
                    prev = per_map.get(cand_entity)
                    if prev is None or sc > prev[0]:
                        per_map[cand_entity] = (sc, rel_actual, direction)

                ranked = sorted([(v[0], cand, v[1], v[2]) for cand, v in per_map.items()], key=lambda x: -x[0])

                if verbose:
                    print(f"  ranked for explicit '{e}': {ranked[:k1]}")

                taken = 0
                for sc, cand, rel_actual, direction in ranked:
                    if taken >= k1:
                        break
                    if cand not in selected_candidates_set:
                        selected_candidates_set.add(cand)
                        selected_candidates_ordered.append(cand)
                    # build triplet according to direction
                    added = False
                    if direction == "forward":
                        if cand in neighbors.get(e, {}).get(rel_actual, []):
                            selected_triplets.append(f"<e>{e}</e>||{rel_actual}||<e>{cand}</e>")
                            added = True
                        elif e in neighbors.get(cand, {}).get(rel_actual, []):
                            selected_triplets.append(f"<e>{cand}</e>||{rel_actual}||<e>{e}</e>")
                            added = True
                    else:  # reverse
                        if e in neighbors.get(cand, {}).get(rel_actual, []):
                            selected_triplets.append(f"<e>{cand}</e>||{rel_actual}||<e>{e}</e>")
                            added = True
                        elif cand in neighbors.get(e, {}).get(rel_actual, []):
                            selected_triplets.append(f"<e>{e}</e>||{rel_actual}||<e>{cand}</e>")
                            added = True
                    # we accept even if added False (it means candidate doesn't actually connect in neighbors) — but prefer only real ones
                    taken += 1

            # deduplicate selected_triplets preserving order
            seen = set()
            dedup_triplets = []
            for t in selected_triplets:
                if t not in seen:
                    dedup_triplets.append(t); seen.add(t)

            # add complete triplets (k2)
            complete_triplets = self._retrieve_complete_triplets(neighbors, pseudo_graph, k2=k2)
            for t in complete_triplets:
                if t not in seen:
                    dedup_triplets.append(t); seen.add(t)

            # Add KG facts for all explicit entities (force-include)
            for e in explicit_entities:
                for rel, vals in neighbors.get(e, {}).items():
                    for v in vals:
                        trip = f"<e>{e}</e>||{rel}||<e>{v}</e>"
                        if trip not in seen:
                            dedup_triplets.append(trip); seen.add(trip)

            per_group_results.append({
                "triplets": dedup_triplets,
                "candidate_nodes_after_top_k": selected_candidates_ordered,
                "explicit_nodes": explicit_entities
            })

        # ---- MERGE all groups to single consolidated output to avoid losing groups ----
        merged_triplets = []
        merged_candidates = set()
        merged_explicit = set()
        seen_all = set()
        for res in per_group_results:
            for t in res.get("triplets", []):
                if t not in seen_all:
                    merged_triplets.append(t); seen_all.add(t)
            for c in res.get("candidate_nodes_after_top_k", []):
                merged_candidates.add(c)
            for e in res.get("explicit_nodes", []):
                merged_explicit.add(e)

        return [{
            "triplets": merged_triplets,
            "candidate_nodes_after_top_k": list(merged_candidates),
            "explicit_nodes": list(merged_explicit)
        }]

    # -------------------------
    # Null-rel groups (frequency)
    # -------------------------
    def resolve_unknown_without_relation(
        self,
        neighbors: Dict[str, Dict[str, List[str]]],
        pseudo_graph: Dict[str, Any],
        k1: int = 3
    ) -> List[Dict[str, Any]]:
        incomplete_groups = pseudo_graph.get("incomplete_groups", {})
        results: List[Dict[str, Any]] = []
        for unk_id, group in incomplete_groups.items():
            explicit_entities = group.get("explicit_entities", [])
            pseudo_relations = group.get("relations", [])

            if not all(r.startswith("null_relation") for r in pseudo_relations):
                continue

            candidate_list: List[str] = []
            for e in explicit_entities:
                if e not in neighbors:
                    continue
                for rel, vals in neighbors[e].items():
                    for v in vals:
                        candidate_list.append(v)

            if not candidate_list:
                results.append({
                    "triplets": [],
                    "candidate_nodes_after_top_k": [],
                    "explicit_nodes": explicit_entities
                })
                continue

            freq = Counter(candidate_list)
            top_candidates = [c for c, _ in freq.most_common(k1)]

            S = []
            for e in explicit_entities:
                if e not in neighbors:
                    continue
                for rel, vals in neighbors[e].items():
                    for v in vals:
                        if v in top_candidates:
                            S.append(f"<e>{e}</e>||{rel}||<e>{v}</e>")

            S = list(dict.fromkeys(S))
            results.append({
                "triplets": S,
                "candidate_nodes_after_top_k": top_candidates,
                "explicit_nodes": explicit_entities
            })
        return results

    # -------------------------
    # Union procedure
    # -------------------------
    def retrive_and_union(self, standardized_triplets: List[str], group_n_decomposed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Top-level orchestration:
         - Extract explicit entities from standardized_triplets and group_n_decomposed
         - retrieve neighbors for them
         - resolve unknowns
         - union triplets (using union_triplets)
         - force-include KG facts for all explicit entities into the final result
        """
        from utils.parser import str_to_triplet, union_triplets

        # 1) collect entities from standardized_triplets
        entities = set()
        for triplet in standardized_triplets:
            try:
                head, rel, tail = str_to_triplet(triplet)
            except Exception:
                continue
            if not head.startswith("unknown_"):
                entities.add(head)
            if not tail.startswith("unknown_"):
                entities.add(tail)

        # 2) collect from group_n_decomposed (complete & incomplete groups)
        for t in group_n_decomposed.get("complete_triplets", []):
            try:
                h, r, ta = self._parse_triplet_str(t) if isinstance(t, str) else (t[0], t[1], t[2])
                if not h.startswith("unknown_"): entities.add(h)
                if not ta.startswith("unknown_"): entities.add(ta)
            except Exception:
                pass

        for gid, g in group_n_decomposed.get("incomplete_groups", {}).items():
            for e in g.get("explicit_entities", []):
                entities.add(e)

        # Retrieve neighbors for all collected explicit entities
        neighbors = self.retrieve_neighbors(list(entities))

        # Resolve unknown groups
        filled_with_relations = self.resolve_unknown_with_relation(
            neighbors=neighbors,
            pseudo_graph=group_n_decomposed,
        )

        filled_without_relations = self.resolve_unknown_without_relation(
            neighbors=neighbors,
            pseudo_graph=group_n_decomposed,
        )

        # Union using provided helper
        unified_results = union_triplets(filled_with_rel=filled_with_relations, filled_without_rel=filled_without_relations)

        # Ensure unified_results exists and has at least one slot
        if not unified_results:
            unified_results = [{"triplets": [], "candidate_nodes_after_top_k": [], "explicit_nodes": list(entities)}]

        # Force-include KG facts for all explicit entities so nothing is lost
        extra_facts = []
        for e in entities:
            for rel, vals in neighbors.get(e, {}).items():
                for v in vals:
                    extra_facts.append(f"<e>{e}</e>||{rel}||<e>{v}</e>")

        # Merge extras into unified_results[0]["triplets"] deduplicated
        seen = set(unified_results[0].get("triplets", []))
        for t in extra_facts:
            if t not in seen:
                unified_results[0].setdefault("triplets", []).append(t)
                seen.add(t)

        return unified_results

    # -------------------------
    # Helper / future: fill missing explicit nodes
    # -------------------------
    def fill_missing_explicit_entities(self, unified_results: List[Dict[str, Any]], pseudo_graph: Dict[str, Any]) -> None:
        return None
