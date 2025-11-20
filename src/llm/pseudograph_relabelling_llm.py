import os
from google import genai
from google.genai import types
from typing import Union

class PseudoGraphRelabellingLLM:
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv('PSEUDOGRAPH_RELABELLING_API_KEY'))

    def submit(self, claim_data: str, entities: Union[str, list], evidence: Union[str, dict], max_tokens: int = 32) -> str:

        if isinstance(entities, list):
            entities = str(entities)
        if isinstance(evidence, dict):
            evidence = str(evidence)

        # Tạo prompt đúng định dạng trong paper ClaimPKG
        prompt = f"""
                Given the following JSON claim data:
                \"{claim_data}\"

                Given the following entities as an array:
                {entities}

                Given the following evidences, each key-value is a pair of enitiy and its relationships:
                {evidence}

                You are now have to find the correct triplet representation of the claim in the format, balancing claim data and evidence:
                <e>HEAD</e> || RELATION || <e>TAIL</e>

                You can skip if there is some issues with the the equivalent evidence. If there are any implicit information in the claim data, use the unknown_i, where is an integer starting from 0.

                For example:
                Claim data: "The author of 'Romeo and Juliet' is from a country that is in Europe."
                Entities: ["Romeo and Juliet", "Europe"]
                Evidence: {
                    "Romeo and Juliet": ["written_by"],
                    "Europe": ["continent_of"]
                }

                Correct triplet: <e>Romeo and Juliet</e> || written_by || <e>unknown_0</e>; <e>unknown_0</e> || continent_of || <e>Europe</e>
                """
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=max_tokens,
                system_instruction="You are a tool to generate triplet representation from claim data, entities, and evidence. With the given information, you have to generate the correct triplet representation balancing claim data and evidence. If there are any relations with prefix of ~, it's a reverse relation (not the negation).",
            )
        )

        if not response or not response.text:
            raise ValueError("No response from the LLM model.")

        return response.text.strip()
