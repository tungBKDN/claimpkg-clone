from html import entities
import os
import sys
from google import genai
from google.genai import types
from typing import Union

from prompt_toolkit import prompt


class PseudoGraphRelabellingLLM:
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    def __init__(self, model_name: str = "gemini-flash-latest", key = None):
        if key == None:
            key = os.getenv('PSEUDOGRAPH_RELABELLING_API_KEY')
        self.CLIENT = genai.Client(api_key=key)
        self.MODEL = model_name
        self.CONFIG = types.GenerateContentConfig(
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_budget=8000),
            system_instruction=[
                types.Part.from_text(text="""You are a tool to generate triplet representation from claim data, entities, and evidence. With the given information, you have to generate the correct triplet representation balancing claim data and evidence. If there are any relations with prefix of ~, it's a reverse relation (not the negation)."""),
            ],
        )

    def generate_prompt(self, claim: str, claim_data: dict) -> str:
        entities: list = claim_data.get("Entity_set", [])
        evidence: dict = claim_data.get("Evidence", {})

        prompt = (
            f"Given the following JSON claim data:\n"
            f"\"{claim}\"\n\n"
            f"Given the following entities as an array:\n"
            f"{str(entities)}\n\n"
            f"Given the following evidences, each key-value is a pair of entity and its relationships:\n"
            f"{str(evidence)}\n\n"
            f"You are now have to find the correct triplet representation of the claim in the format, "
            f"balancing claim data and evidence:\n"
            f"<e>HEAD</e> || RELATION || <e>TAIL</e>\n\n"
            f"You can skip if there is some issues with the equivalent evidence. "
            f"If there are any implicit information in the claim data, use the unknown_i, "
            f"where is an integer starting from 0. And some abundant relationships can be ignored based on the main context of the claim. And must choose the correct relationships if you are pooling from, like keeping the same timeline, etc.\n\n"
            f"For example:\n"
            f"Claim data: \"The author of 'Romeo and Juliet' is from a country that is in Europe.\"\n"
            f"Entities: [\"Romeo and Juliet\", \"Europe\"]\n"
            f"Evidence: {{\n"
            f"    \"Romeo and Juliet\": [\"written_by\"],\n"
            f"    \"Europe\": [\"continent_of\"]\n"
            f"}}\n\n"
            f"Correct triplet: <e>Romeo and Juliet</e> || written_by || <e>unknown_0</e>; "
            f"<e>unknown_0</e> || continent_of || <e>Europe</e>\n"
        )
        return prompt

    def submit(self, claim: str, claim_data: dict) -> str:
        # Tạo prompt đúng định dạng trong paper ClaimPKG
        prompt = self.generate_prompt(claim, claim_data)

        response = self.CLIENT.models.generate_content(
            model=self.MODEL,
            contents=prompt,
            config=self.CONFIG,
        )

        if not response or not response.text:
            raise ValueError("No response from the LLM model.")

        return response.text.strip()