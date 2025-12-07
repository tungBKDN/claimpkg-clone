import json
import os
from google import genai
from google.genai import types

class GeneralLLM:
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv('GENERAL_LLM_API_KEY'))

    def submit(self, claim: str, graph_string: str, max_tokens: int = 2048) -> dict:
        # Tạo prompt đúng định dạng trong paper ClaimPKG
        prompt = f"""
                    Claim: {claim}

                    Evidence:
                    {graph_string.strip()}

                    Question: Is the claim supported by the evidence?
                    Please answer with one of [Supported, Refuted, NotEnoughInfo]
                    and give a short explanation in one sentence.

                    You should answer with one of [Supported, Refuted, NotEnoughInfo] and give a short explanation in one sentence. The output must be in the format: of JSON object like this:
                    {{"verdict": "Supported/Refuted/NotEnoughInfo", "explanation": "your explanation here"}}.
                    - Do not add any extra text outside the JSON object like ``` or ```json.
                    - Start with {{ and end with }}.
                    """
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=max_tokens,
                system_instruction="""You are a fact checker. You are going to receive a claim, and evidence as a graph in text format of triples. You need to determine whether the claim is supported by the evidence, refuted by the evidence, or there is not enough information in the evidence to determine whether the claim is true or false. If a relationship has prefix of ~, it means the negation of that relationship. You should answer with one of [Supported, Refuted, NotEnoughInfo] and give a short explanation in one sentence. The output must be in the format: of JSON object like this:
{{"verdict": "Supported/Refuted/NotEnoughInfo", "explanation": "your explanation here"}}.
                - Do not add any extra text outside the JSON object like ``` or ```json.
                - Start with { and end with }.
                """
            )
        )

        if not response or not response.text:
            raise ValueError("No response from the LLM model.")

        # Parse the json as dict
        result_json = {
            "verdict": "",
            "explanation": "",
            "final_graph": ""
        }

        import re
        # Remove any thing before the first { and after the last }
        response_text = re.search(r"\{.*\}", response.text, re.DOTALL).group()

        # parse text to get the json part only
        response_text = response_text.strip()
        try:
            _json = json.loads(response_text)
            result_json["verdict"] = _json.get("verdict", "")
            result_json["explanation"] = _json.get("explanation", "")
            result_json["final_graph"] = graph_string.strip()
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from LLM response: {response_text}")

        return result_json