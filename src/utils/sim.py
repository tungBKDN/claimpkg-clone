import torch.nn.functional as F
import torch
from typing import Tuple, List, Dict, Callable
from heapq import nlargest

class Similarity:
    def __init__(self, encoder_model: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize the Similarity class with a specified encoder model.
        Parameters:
            encoder_model (str): The name of the pretrained model to use for encoding.
        """
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(encoder_model)

    def sim(self, r1: str, r2: str) -> float:
        """
        Compute similarity between two relations using the initialized encoder.
        Parameters:
            r1 (str): The first relation string.
            r2 (str): The second relation string.
        Returns:
            float: The cosine similarity score between the two relations.

        Example:
        sim.sim("birth place", "place of birth"): return: 0.92
        """

        embeddings = self.encoder.encode([r1, r2], convert_to_tensor=True, normalize_embeddings=True)
        sim = F.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
        return sim.item()

    def batch_sim(self, query: str, top_k: int = 5, candidates: list = []) -> list[Tuple[str, float]]:
        """
        Compute the top-k most similar candidates to the query relation.
        Parameters:
            query (str): The query relation string.
            top_k (int): The number of top similar candidates to return.
            candidates (list): A list of candidate relation strings.

        Returns:
            list: A list of the top-k most similar candidate relations.

        Exceptions:
            ValueError: If the candidates list is empty.

        Example:
        ```
        >>> sim.batch_sim("birth", top_k=3, candidates=['place_of_death', 'date_of_death', 'death_place', 'death_year', 'country_of_death', etc]):
        [('birth_year', 0.7336347103118896), ('country_of_birth', 0.7212591171264648), ('year_of_birth', 0.7191925644874573)]
        ```
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty.")

        kg_embeds = self.encoder.encode(candidates, normalize_embeddings=True)
        query_embed = self.encoder.encode([query], normalize_embeddings=True)
        scores = query_embed @ kg_embeds.T
        top_k_indices = torch.topk(torch.tensor(scores), top_k).indices.tolist()[0]
        return [(candidates[i], scores[0][i].item()) for i in top_k_indices]

    def score(
        self,
        candidate_entity: str,
        explicit_entities: List[str],
        pseudo_relations: List[str],
        KG: Dict[str, List[Tuple[str, str]]],
        sim_func: Callable[[str, str], float],
        normalize: bool = True
    ) -> float:
        """
        Compute the semantic matching score of a candidate entity `candidate_entity`
        for resolving an unknown node in a pseudo-subgraph, following Eq. (5) in ClaimPKG.

        The score is computed by summing the similarity between the pseudo-relations
        (relations connected to the unknown entity) and the actual relations in the KG
        that link the candidate to explicit entities.

        Parameters
        ----------
        candidate_entity : str
            The entity in the KG being evaluated as a possible replacement for an unknown node.
        explicit_entities : List[str]
            A list of known (explicit) entities connected to the unknown in the pseudo-subgraph.
        pseudo_relations : List[str]
            The corresponding relations between the unknown and each explicit entity.
        KG : Dict[str, List[Tuple[str, str]]]
            The knowledge graph, represented as a dictionary:
            { head_entity: [(relation, tail_entity), ...], ... }.
        sim_func : Callable[[str, str], float]
            Function computing similarity between two relation strings (e.g., embedding cosine similarity).
        normalize : bool, optional
            Whether to normalize the final score by number of relations, default=True.

        Returns
        -------
        float
            The cumulative similarity score representing how well the candidate matches
            the pseudo-relations and connects to the explicit entities.
        """
        total_score = 0.0
        match_count = 0

        for e_ui, r_ui in zip(explicit_entities, pseudo_relations):
            kg_edges = KG.get(e_ui, [])
            for r, tail in kg_edges:
                if tail == candidate_entity:
                    sim_val = sim_func(r_ui, r)
                    total_score += sim_val
                    match_count += 1

        if normalize and match_count > 0:
            total_score /= match_count

        return total_score

    def rank_candidates(
        self,
        candidate_sets: List[List[str]],
        explicit_entities: List[str],
        pseudo_relations: List[str],
        KG: Dict[str, List[Tuple[str, str]]],
        sim_func: Callable[[str, str], float],
        k1: int = 3,
        normalize: bool = True,
        aggregate: str = "max"
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate entities based on their relevance to the unknown entity group
        using the scoring mechanism defined in Eq. (5)-(6) of ClaimPKG.

        Parameters
        ----------
        candidate_sets : List[List[str]]
            A list of candidate lists, each corresponding to one explicit entity e_ui.
            For example: [[cand1, cand2], [cand3, cand4]].
        explicit_entities : List[str]
            Entities directly connected to the unknown entity in the pseudo-subgraph.
        pseudo_relations : List[str]
            Relations corresponding to each explicit entity.
        KG : Dict[str, List[Tuple[str, str]]]
            The knowledge graph data structure.
        sim_func : Callable[[str, str], float]
            Function measuring similarity between two relations.
        k1 : int, optional
            Number of top candidates to select (default=3).
        normalize : bool, optional
            Whether to normalize the score of each candidate (default=True).
        aggregate : str, optional
            Aggregation strategy for merging candidate scores from multiple sets.
            Options:
            - "max"  : keep the maximum score per entity
            - "mean" : average over occurrences
            - "sum"  : sum over occurrences

        Returns
        -------
        List[Tuple[str, float]]
            A list of tuples (candidate_entity, score), sorted descending by score.
        """
        scored = {}

        # Evaluate all candidates from each set
        for candidates in candidate_sets:
            for c in candidates:
                s = self.score(c, explicit_entities, pseudo_relations, KG, sim_func, normalize)
                if c not in scored:
                    scored[c] = [s]
                else:
                    scored[c].append(s)

        # Aggregate scores from multiple occurrences
        aggregated_scores = {}
        for c, vals in scored.items():
            if aggregate == "max":
                aggregated_scores[c] = max(vals)
            elif aggregate == "mean":
                aggregated_scores[c] = sum(vals) / len(vals)
            elif aggregate == "sum":
                aggregated_scores[c] = sum(vals)
            else:
                raise ValueError(f"Unknown aggregate mode: {aggregate}")

        # Select top-k1 highest scoring candidates
        topk = nlargest(k1, aggregated_scores.items(), key=lambda x: x[1])

        return topk
