import os
from google import genai
from google.genai import types

class GeneralLLM:
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv('GENERAL_LLM_API_KEY'))

    def submit(self, claim: str, graph_string: str, max_tokens: int = 256) -> str:
        # Tạo prompt đúng định dạng trong paper ClaimPKG
        prompt = f"""
                    Claim: {claim}

                    Evidence:
                    {graph_string.strip()}

                    Question: Is the claim supported by the evidence?
                    Please answer with one of [Supported, Refuted, NotEnoughInfo]
                    and give a short explanation in one sentence.
                    """
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=max_tokens,
                system_instruction="You are a fact checker. You are going to receive a claim, and evidence as a graph in text format of triples. You need to determine whether the claim is supported by the evidence, refuted by the evidence, or there is not enough information in the evidence to determine whether the claim is true or false. If a relationship has prefix of ~, it means the negation of that relationship. You should answer with one of [Supported, Refuted, NotEnoughInfo] and give a short explanation in one sentence.",
            )
        )

        if not response or not response.text:
            raise ValueError("No response from the LLM model.")

        return response.text.strip()
