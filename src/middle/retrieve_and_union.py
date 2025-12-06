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
        Additionally builds a reverse-index in-memory so we can lookup entities that point to a literal value.
        Returns the forward neighbor map only (reverse map computed in _build_reverse_neighbors()).
        """
        # Primary forward neighbors from connector (assumed format provided by you)
        forward = self.kg_connector.get_neighbors(entity_names)
        # Ensure keys exist for all requested entity_names (even if empty)
        for e in entity_names:
            if e not in forward:
                forward[e] = {}
        return forward

    def _build_reverse_neighbors(self, neighbors: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Build reverse mapping: for each triple (src -rel-> tgt) in neighbors,
        add reverse[tgt][rel].append(src). This allows handling explicit literal nodes
        or nodes that are only present as values.
        """
        rev = defaultdict(dict)  # type: Dict[str, Dict[str, List[str]]]
        for src, relmap in neighbors.items():
            for rel, vals in relmap.items():
                for v in vals:
                    if rel not in rev[v]:
                        rev[v][rel] = []
                    rev[v][rel].append(src)
        return dict(rev)

    def _build_local_KG(self, neighbors: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Build a local adjacency-like structure suitable for scoring:
        { head_entity: [(relation, tail_entity), ...], ... }
        Works by reading forward neighbors only.
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
        """
        Parse strings like "<e>Head</e>||relation||<e>Tail</e>" or "Head||relation||Tail".
        Returns (head, relation, tail) with entity tags removed.
        """
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
    ) -> float:
        """
        Preferred scoring: if self.sim.score exists, use it (it already implements a good variant).
        Fallback: compute sum(sim(r_ui, r_actual)) across KG edges linking explicit_entities -> candidate.
        """
        # If Similarity exposes a 'score' function matching your provided signature, call it.
        if hasattr(self.sim, "score"):
            try:
                # normalize=True consistent with the signature you provided earlier
                return float(self.sim.score(
                    candidate_entity=candidate,
                    explicit_entities=explicit_entities,
                    pseudo_relations=pseudo_relations,
                    KG=KG,
                    sim_func=self.sim.sim,
                    normalize=True
                ))
            except Exception:
                # fallback to pairwise
                pass

        # Fallback implementation (sum of sim over matching KG edges)
        total = 0.0
        match_count = 0
        for e_ui, r_ui in zip(explicit_entities, pseudo_relations):
            # KG.get may be empty if e_ui not found; that's fine
            kg_edges = KG.get(e_ui, [])
            for r_actual, tail in kg_edges:
                if tail == candidate:
                    try:
                        total += float(self.sim.sim(r_ui, r_actual))
                    except Exception:
                        total += 0.0
                    match_count += 1
        if match_count > 0:
            # In the paper Eq(5) sums; some implementations normalize; we use normalize=True behaviour to be stable
            return total / match_count
        return 0.0

    # -------------------------
    # _retrieve_complete_triplets (k2)
    # -------------------------
    def _retrieve_complete_triplets(
        self,
        neighbors: Dict[str, Dict[str, List[str]]],
        pseudo_graph: Dict[str, Any],
        k2: int = 1
    ) -> List[str]:
        """
        For pseudo complete triplets (both head & tail explicit), find up to k2 actual relations between them.
        Returns formatted triplet strings.
        """
        results: List[str] = []
        KG = self._build_local_KG(neighbors)

        # collect pseudo complete triples
        triples: List[Tuple[str, str, str]] = []
        if "complete_triplets" in pseudo_graph and pseudo_graph["complete_triplets"]:
            # might be strings or tuples
            for t in pseudo_graph["complete_triplets"]:
                if isinstance(t, str):
                    try:
                        triples.append(self._parse_triplet_str(t))
                    except Exception:
                        continue
                elif isinstance(t, (list, tuple)) and len(t) == 3:
                    triples.append((t[0], t[1], t[2]))
        else:
            # scan raw triplets if any
            for t in pseudo_graph.get("triplets", []):
                try:
                    head, rel, tail = self._parse_triplet_str(t)
                    if not head.startswith("unknown_") and not tail.startswith("unknown_"):
                        triples.append((head, rel, tail))
                except Exception:
                    continue

        for head, rel_pseudo, tail in triples:
            # find relations in neighbors[head] that lead to tail
            rels_found: List[Tuple[float, str]] = []
            for rel_actual, vals in neighbors.get(head, {}).items():
                if tail in vals:
                    try:
                        score = float(self.sim.sim(rel_pseudo, rel_actual))
                    except Exception:
                        score = 0.0
                    rels_found.append((score, rel_actual))
            if not rels_found:
                # no direct relation; the paper suggests decomposition; we skip here
                continue
            rels_found.sort(key=lambda x: x[0], reverse=True)
            for score, rel_actual in rels_found[:k2]:
                results.append(f"<e>{head}</e>||{rel_actual}||<e>{tail}</e>")

        # deduplicate while preserving order
        seen = set()
        deduped = []
        for r in results:
            if r not in seen:
                deduped.append(r)
                seen.add(r)
        return deduped

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
        verbose: bool = True
    ) -> List[Dict]:
        """
        Rewritten retrieval with detailed debug and options.

        - normalize_scores: if True, call self.sim.score(..., normalize=True) or average fallback;
                           if False, use raw sum (closer to paper Eq.(5)).
        - verbose: print internal candidate lists, scores and selection steps.
        - k1: top per explicit entity (increase if you want more coverage)
        """
        incomplete_groups = pseudo_graph.get("incomplete_groups", {})
        results = []
        KG = self._build_local_KG(neighbors)
        reverse_neighbors = self._build_reverse_neighbors(neighbors)

        for unk_id, group in incomplete_groups.items():
            explicit_entities = group.get("explicit_entities", [])
            pseudo_relations  = group.get("relations", [])

            # Skip null-rel groups (paper handles separately)
            if all(r.startswith("null_relation") for r in pseudo_relations):
                if verbose:
                    print(f"[resolve] skipping {unk_id} because all relations are null_relation")
                continue

            if verbose:
                print(f"\n[resolve] Processing group {unk_id}")
                print(" explicit_entities:", explicit_entities)
                print(" pseudo_relations:", pseudo_relations)

            # -------------------------
            # Build per-entity candidate lists (preserve per-entity grouping)
            # include forward and reverse hits
            # -------------------------
            per_entity_candidates = []
            for e in explicit_entities:
                cand_pairs = []  # (candidate_entity, rel_actual, direction) direction 'forward' means e->cand
                # forward
                if e in neighbors:
                    for rel_actual, vals in neighbors[e].items():
                        for v in vals:
                            cand_pairs.append((v, rel_actual, "forward"))
                # reverse (e is a target value, find sources that point to e)
                if e in reverse_neighbors:
                    for rel_actual, sources in reverse_neighbors[e].items():
                        for src in sources:
                            cand_pairs.append((src, rel_actual, "reverse"))
                per_entity_candidates.append(cand_pairs)

            if verbose:
                for i, e in enumerate(explicit_entities):
                    print(f"  candidates for explicit[{i}] = '{e}':")
                    for cand_entity, rel_actual, direction in per_entity_candidates[i]:
                        print(f"    - {cand_entity} via {rel_actual} ({direction})")

            # If every per_entity candidate list empty → append empty result and continue
            if all(len(c) == 0 for c in per_entity_candidates):
                if verbose:
                    print("  -> All candidate lists empty for this group; appending empty result.")
                results.append({
                    "triplets": [],
                    "candidate_nodes_after_top_k": [],
                    "explicit_nodes": explicit_entities
                })
                continue

            # -------------------------
            # Build set of all candidate entities
            # -------------------------
            all_candidates = set()
            for cand_list in per_entity_candidates:
                for cand_entity, _, _ in cand_list:
                    all_candidates.add(cand_entity)

            if verbose:
                print("  all unique candidates:", all_candidates)

            # -------------------------
            # Compute global scores per Eq.(5) (use self.sim.score if available)
            # -------------------------
            candidate_scores = {}
            for cand in all_candidates:
                # Use self.sim.score if available with normalize param; else fallback
                try:
                    if hasattr(self.sim, "score"):
                        score_val = float(self.sim.score(
                            candidate_entity=cand,
                            explicit_entities=explicit_entities,
                            pseudo_relations=pseudo_relations,
                            KG=KG,
                            sim_func=self.sim.sim,
                            normalize=normalize_scores
                        ))
                    else:
                        # fallback compute raw sum (or average if normalize_scores True)
                        total = 0.0
                        matches = 0
                        for e_ui, r_ui in zip(explicit_entities, pseudo_relations):
                            for (r_act, nb) in KG.get(e_ui, []):
                                if nb == cand:
                                    total += float(self.sim.sim(r_ui, r_act))
                                    matches += 1
                        score_val = (total / matches) if (normalize_scores and matches>0) else total
                except Exception as ex:
                    if verbose:
                        print("   [warn] score computation failed for", cand, ex)
                    score_val = 0.0
                candidate_scores[cand] = score_val

            if verbose:
                print("  candidate_scores:")
                for c, sc in sorted(candidate_scores.items(), key=lambda x: -x[1]):
                    print(f"    {c} : {sc:.6f}")

            # -------------------------
            # Per-explicit top-k1 selection (Eq.6) using candidate_scores as global metric
            # -------------------------
            selected_candidates = []
            selected_candidates_set = set()
            selected_triplets = []

            for idx, e in enumerate(explicit_entities):
                cand_pairs = per_entity_candidates[idx]
                if not cand_pairs:
                    if verbose:
                        print(f"  explicit '{e}' has no candidates")
                    continue

                # deduplicate cand_pairs by entity while keeping best rel/direction (choose highest candidate_scores)
                per_map = {}
                for cand_entity, rel_actual, direction in cand_pairs:
                    sc = candidate_scores.get(cand_entity, 0.0)
                    # keep the rel_actual/direction that yields highest sc (simple heuristic)
                    prev = per_map.get(cand_entity)
                    if prev is None or sc > prev[0]:
                        per_map[cand_entity] = (sc, rel_actual, direction)

                # build ranked list by global score
                ranked = sorted([(v[0], cand, v[1], v[2]) for cand, v in per_map.items()], key=lambda x: -x[0])

                if verbose:
                    print(f"  ranked candidates for explicit '{e}':")
                    for sc, cand, rel_actual, direction in ranked[:k1]:
                        print(f"    {cand} ({rel_actual}, {direction}) score={sc:.6f}")

                taken = 0
                for sc, cand, rel_actual, direction in ranked:
                    if taken >= k1:
                        break
                    # add candidate
                    if cand not in selected_candidates_set:
                        selected_candidates_set.add(cand)
                        selected_candidates.append(cand)
                    # build triplet according to direction:
                    if direction == "forward":
                        # e -> rel_actual -> cand (if exists)
                        if cand in neighbors.get(e, {}).get(rel_actual, []):
                            selected_triplets.append(f"<e>{e}</e>||{rel_actual}||<e>{cand}</e>")
                        else:
                            # fallback try candidate -> rel_actual -> e
                            if e in neighbors.get(cand, {}).get(rel_actual, []):
                                selected_triplets.append(f"<e>{cand}</e>||{rel_actual}||<e>{e}</e>")
                    else:  # reverse direction (cand -> rel_actual -> e)
                        if e in neighbors.get(cand, {}).get(rel_actual, []):
                            selected_triplets.append(f"<e>{cand}</e>||{rel_actual}||<e>{e}</e>")
                        else:
                            # fallback try e -> rel_actual -> cand
                            if cand in neighbors.get(e, {}).get(rel_actual, []):
                                selected_triplets.append(f"<e>{e}</e>||{rel_actual}||<e>{cand}</e>")
                    taken += 1

            # deduplicate selected_triplets preserving order
            seen = set()
            dedup_triplets = []
            for t in selected_triplets:
                if t not in seen:
                    dedup_triplets.append(t)
                    seen.add(t)

            if verbose:
                print("  selected_candidates_ordered:", selected_candidates)
                print("  produced triplets:")
                for t in dedup_triplets:
                    print("   ", t)

            # -------------------------
            # handle complete triplets (k2)
            # -------------------------
            complete_triplets = self._retrieve_complete_triplets(neighbors, pseudo_graph, k2=k2)
            for t in complete_triplets:
                if t not in seen:
                    dedup_triplets.append(t)
                    seen.add(t)

            # --- NEW: add KG facts for any explicit entity that exists in KG adjacency ---
            for e in explicit_entities:
                if e in neighbors:  # neighbors = adjacency built from KG
                    for rel, vals in neighbors[e].items():
                        for v in vals:
                            trip = f"<e>{e}</e>||{rel}||<e>{v}</e>"
                            if trip not in seen:
                                dedup_triplets.append(trip)
                                seen.add(trip)

            # done


            results.append({
                "triplets": dedup_triplets,
                "candidate_nodes_after_top_k": selected_candidates,
                "explicit_nodes": explicit_entities
            })

        return results

    # -------------------------
    # Null-rel groups (frequency)
    # -------------------------
    def resolve_unknown_without_relation(
        self,
        neighbors: Dict[str, Dict[str, List[str]]],
        pseudo_graph: Dict[str, Any],
        k1: int = 3
    ) -> List[Dict[str, Any]]:
        """
        For groups where all relations are null_relation_X: use frequency-based scoring per ClaimPKG.
        """
        incomplete_groups = pseudo_graph.get("incomplete_groups", {})
        results: List[Dict[str, Any]] = []
        for unk_id, group in incomplete_groups.items():
            explicit_entities = group.get("explicit_entities", [])
            pseudo_relations = group.get("relations", [])

            if not all(r.startswith("null_relation") for r in pseudo_relations):
                continue

            candidate_list: List[str] = []
            per_entity_sets: List[List[str]] = []
            for e in explicit_entities:
                if e not in neighbors:
                    per_entity_sets.append([])
                    continue
                cand = []
                for rel, vals in neighbors[e].items():
                    for v in vals:
                        cand.append(v)
                per_entity_sets.append(cand)
                candidate_list.extend(cand)

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
            for i, e in enumerate(explicit_entities):
                if e not in neighbors:
                    continue
                for rel, vals in neighbors[e].items():
                    for v in vals:
                        if v in top_candidates:
                            S.append(f"<e>{e}</e>||{rel}||<e>{v}</e>")

            # dedup
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
         - Extract non-unknown entities from standardized_triplets
         - retrieve neighbors (forward) for them
         - resolve unknowns with relations and without relations
         - perform union (caller provides union_triplets function)
        Returns unified_results (as returned by union_triplets).
        """
        entities = set()
        from utils.parser import str_to_triplet, union_triplets  # assume these exist in your project
        for triplet in standardized_triplets:
            head, rel, tail = str_to_triplet(triplet)
            if not head.startswith("unknown_"):
                entities.add(head)
            if not tail.startswith("unknown_"):
                entities.add(tail)

        neighbors = self.retrieve_neighbors(list(entities))
        # Note: we do NOT remove keys with quotes etc. Keep as-is.

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

    # -------------------------
    # Helper / future: fill missing explicit nodes
    # -------------------------
    def fill_missing_explicit_entities(self, unified_results: List[Dict[str, Any]], pseudo_graph: Dict[str, Any]) -> None:
        """
        Placeholder to implement any post-processing needed to fill missing explicit entity slots.
        Kept for API compatibility.
        """
        return None
