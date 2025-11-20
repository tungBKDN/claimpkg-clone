import os
from google import genai
from google.genai import types


class PseudoGraphCheckingLLM:
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv('PSEUDOGRAPH_CHECKING_API_KEY'))

    def submit(self, claim_data: str, graph_string: str, max_tokens: int = 32) -> str:
        # Tạo prompt đúng định dạng trong paper ClaimPKG
        prompt = f"""
                Given the following JSON claim data:
                {claim_data}

                And the following triplets in the format:
                <e>HEAD</e> || RELATION || <e>TAIL</e>
                {graph_string}

                Rules:
                1. 'Entity_set' lists entities explicitly mentioned in the claim.
                2. 'Evidence' maps each entity to one or more relation paths.
                - A relation 'r' means HEAD --r--> TAIL.
                - '~r' means TAIL --r--> HEAD.
                3. Multi-hop paths must be broken into correct hop-by-hop triplets.
                4. Hidden or implicit entities must be represented as unknown_i.
                5. Triplets must match the structure implied by Evidence.
                6. Only answer the correctness of the triplets, not the claim.

                Your task:
                Determine whether the triplets are consistent with the given Evidence.
                Answer strictly with one of:
                - CORRECT: triplets match Evidence structure
                - INCORRECT: triplets disagree with Evidence
                - DATA_PROBLEM: claim_data itself is malformed or contradictory
                """
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=max_tokens,
                system_instruction="You are a pseudo-graph checker. Evaluate whether the generated triplets match the Evidence structure. Output only one token from [CORRECT, INCORRECT, DATA_PROBLEM].",
            )
        )

        if not response or not response.text:
            raise ValueError("No response from the LLM model.")

        return response.text.strip()
