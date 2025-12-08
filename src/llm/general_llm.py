import json
import os
from google import genai
from google.genai import types

class GeneralLLM:
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv('GENERAL_LLM_API_KEY'))

    def submit(self, claim: str, graph_string: str, max_tokens: int = 2048, completed: dict = None) -> dict:
        # Tạo prompt đúng định dạng trong paper ClaimPKG
        prompt = f"""
                    Claim: {claim}

                    Evidence:
                    {graph_string.strip()}

                    Completed triplets:
                    {str(completed)}

                    Question: Is the claim supported by the evidence?
                    Please answer with one of [Supported, Refuted, NotEnoughInfo]
                    and give a short explanation in one sentence.

                    Completed is the triplets that are known to be true., it's prioritied over the evidence graph.

                    You should answer with one of [Supported, Refuted, NotEnoughInfo] and give a short explanation in one sentence. The output must be in the format: of JSON object like this:
                    {{"verdict": "Supported/Refuted/NotEnoughInfo", "explanation": "your explanation here"}}.
                    - Do not add any extra text outside the JSON object like ``` or ```json.
                    - Start with {{ and end with }}.
                    """
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=max_tokens,
                system_instruction="""You are a fact checker. You are going to receive a claim, and evidence as a graph in text format of triples. You need to determine whether the claim is supported by the evidence, refuted by the evidence, or there is not enough information in the evidence to determine whether the claim is true or false. If a relationship has prefix of ~, it means the reverse of that relationship. You should answer with one of [Supported, Refuted, NotEnoughInfo] and give a short explanation in one sentence. The output must be in the format: of JSON object like this:
                - You can try to reverse the the head, tail of a triplet to see if it helps your reasoning. (Eg. <e>A</e>||northwest||<e>B</e>, it can be both A is northwest of B or B is southeast of A due to the duplicity of direction in DBPedia.)
{{"verdict": "Supported/Refuted/NotEnoughInfo", "explanation": "your explanation here"}}.
                - Do not add any extra text outside the JSON object like ``` or ```json.
                - Start with { and end with }.
                - You you should think/reason based on the evidence triples to come to your conclusion. Sometime the clues are not obvious when it comes to reasoning about geography, dates, or common sense knowledge, or some basic facts. (Eg: a's son is a An Nguyen so you can derive that his son is a Vietnamese.)
                """
            )
        )

        if not response or not response.text:
            return {
                "verdict": "NotEnoughInfo",
                "explanation": "Empty response from LLM.",
                "final_graph": graph_string.strip()
            }

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

            # Delete all extra space right and left in || of graph_string
            graph_string = re.sub(r"\s*\|\|\s*", "||", graph_string)
            result_json["final_graph"] = graph_string.strip()
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from LLM response: {response_text}")

        return result_json