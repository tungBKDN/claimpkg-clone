import os
import sys
sys.path.append('..')

from kg_connector.kg_connector import KGConnector
from llm.general_llm import GeneralLLM
from llm.specialized_llm import SpecializedLLM
from llm.basic_sense_llm import BasicSenseLLM
from llm.psedograph_generator_llm import PseudographGeneratorLLM
import utils.parser as par
from middle.group_n_decompose import GroupNDecompose
from middle.retrieve_and_union import RetrieveAndUnion
from embeddings.embedder import Embedder
from utils.sim import Similarity
import re
class Pipeline:
    def __init__(self):
        self.kg_connector = KGConnector()
        self.general_llm = GeneralLLM()
        self.specialized_llm = SpecializedLLM()
        self.sim = Similarity()
        self.embedder = Embedder(kg_connector=self.kg_connector, sim=self.sim)
        self.group_n_decompose = GroupNDecompose(embedder=self.embedder, kg_connector=self.kg_connector)
        self.retrieve_and_union = RetrieveAndUnion(kg_connector=self.kg_connector)
        self.pseudograph_generator = PseudographGeneratorLLM()
        self.basic_sense_llm = BasicSenseLLM()

    def run(self, claim: str, specialize_mode: str = "FEWSHOT", retry: int = 3) -> dict:
        # 0. Using basic sense LLM to check if the claim is valid
        basic_filter_output = self.basic_sense_llm.submit(claim=claim)
        if basic_filter_output["verdict"] == "Refuted" or basic_filter_output["verdict"] == "Supported":
            return basic_filter_output
        elif basic_filter_output["verdict"] == "Unsupported":
            return basic_filter_output
        elif basic_filter_output["verdict"] != "PassedDown":
            raise ValueError("Invalid verdict from BasicSenseLLM.")



        # 1. Get Pseudo Graph from claim
        pseudo_graph_string = ""
        if specialize_mode == "FEWSHOT":
            pseudo_graph_string = self.pseudograph_generator.generate(claim=claim, retry=retry)
        elif specialize_mode == "FINETUNE":
            pseudo_graph_string = self.specialized_llm.generate(input_text=claim)
        # 1.1. Filtering only triplets that exist in the pseudo_graph_string in case the LLM added extra text: get all substring that match the triplet pattern <e>TEXT</e> || TEXT || <e>TEXT</e>
        # Use re
        triplet_pattern = r"<e>.*?</e> \|\| .*? \|\| <e>.*?</e>"
        pseudo_graph_string = re.findall(triplet_pattern, pseudo_graph_string)
        # Remove strings that contains ENTITY inside
        pseudo_graph_string = [triplet for triplet in pseudo_graph_string if "ENTITY" not in triplet]

        print("Generated Pseudo Graph Triplets:")
        for triplet in pseudo_graph_string:
            print(triplet)

        # 2. Group and decompose Pseudo Graph
        grouped_decomposed = self.group_n_decompose.group_n_decompose(triplets=pseudo_graph_string)

        print("Grouped and Decomposed Triplets:")
        print(grouped_decomposed)

        # 3. Retrieve and Union
        unified_triplets = self.retrieve_and_union.retrive_and_union(standardized_triplets=pseudo_graph_string, group_n_decomposed=grouped_decomposed)

        # 4. Passing the triplets to the final LLM to generate the final answer
        final_retrieved_triplets = ""
        for triplet in unified_triplets:
            final_retrieved_triplets += triplet["triplet_as_string"] + "\n"

        final_answer = self.general_llm.submit(claim=claim, graph_string=final_retrieved_triplets)
        return final_answer
