from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
from typing import Tuple

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
