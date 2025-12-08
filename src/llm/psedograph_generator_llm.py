import os
from google import genai
from google.genai import types
import dotenv
from dotenv import load_dotenv
import time
from google.genai.errors import ServerError


FEWSHOT = r"""
You are a tool that converts claims into pseudo-subgraph triplets
following the ClaimPKG paper, entity name conventions should be close to ontology as DBPedia, as well as relations.

RULES:
- Output ONLY triplets.
- Each triplet must be on its own line.
- Absolutely NO explanation, NO JSON, NO comments.
- Entity mentions must use <e>...</e>.
- Implicit entities must be created as unknown_0, unknown_1, ...
- Format each line exactly:
  <e>HEAD</e> || relation || <e>TAIL</e>
  or <e>unknown_i</e> || relation || <e>TAIL</e>
  or <e>HEAD</e> || relation || <e>unknown_i</e>
- You can use "~" to indicate the reversed relation.
- Do NOT hallucinate entities. Only use what is explicit or implied.
- If a relation is mutual, (eg. sibling), output both directions.
- More over, if the claim related to other things (for example: about comparing temperature of two cties that are in two different temperature zones, you can add relations of lattitude, or dimensional you can use directions and add that one more to the pseudo-graph together with other that existed).
- Be flexible in creating instanceOf relations, sometimes it is not explicitly mentioned but it is necessary to add it to the pseudo-graph. Only add instanceOf relations when it is necessary or obviously mentioned.

THESE ARE SOME RELATIONS EXAMPLE AND IT USE CASES:
- foundingDate: the date when an organization was founded, or a country was established, etc.
- league: the league that a sports team participates in.
- vicePresident: the vice president of a country/vice president who works under a president.

====================
FEW-SHOT EXAMPLES
====================

Claim:
"Barack Obama was born in Hawaii."

Triplets:
<e>Barack Obama</e> || birthPlace || <e>Hawaii</e>

--------------------

Claim:
"The founder of Tesla was born in South Africa."

Triplets:
<e>unknown_0</e> || birthPlace || <e>South Africa</e>
<e>Tesla</e> || founder || <e>unknown_0</e>

--------------------

Claim:
"Einstein was not born in Austria."

Triplets:
<e>Einstein</e> || birthPlace || <e>Austria</e>

--------------------

Claim:
"Karl Marx influenced a philosopher who lived in London."

Triplets:
<e>Karl Marx</e> || influenced || <e>unknown_0</e>
<e>unknown_0</e> || residence || <e>London</e>

--------------------

Claim:
"Tác phẩm Aurora được sáng tác ở phía đông nam của phòng thu do Lina Mehta chỉ đạo, có biệt danh là 'Silver Dawn', và được phát hành bởi một hãng được thành lập bởi nghệ sĩ cổ điển Rafael Cruz. Nó cũng ảnh hưởng đến một album nằm phía bắc của Aurora."

Triplets:
"<e>unknown_0</e> || director || <e>Lina Mehta</e>",
"<e>Aurora</e> || southeastOf || <e>unknown_0</e>",
"<e>Aurora</e> || nickname || <e>Silver Dawn</e>",
"<e>unknown_1</e> || foundedBy || <e>Rafael Cruz</e>",
"<e>Aurora</e> || releasedBy || <e>unknown_1</e>",
"<e>Aurora</e> || influences || <e>unknown_2</e>",
"<e>unknown_2</e> || northOf || <e>Aurora</e>"

--------------------

Claim:
"Japan has a higher population than South Korea."

Triplets:
<e>Japan</e> || hasHigherPopulationThan || <e>South Korea</e>
<e>South Korea</e> || hasLowerPopulationThan || <e>Japan</e>
<e>Japan</e> || population || <e>unknown_0</e>
<e>South Korea</e> || population || <e>unknown_1</e>

--------------------

Claim:
"The number of Provinces in Vietnam is greater than the number of States in the USA."

Triplets:
<e>Vietnam</e> || numberOfProvinces || <e>unknown_0</e>
<e>USA</e> || numberOfStates || <e>unknown_1</e>

====================
END OF FEW-SHOT
====================

Now convert the next claim into triplets only.
"""


class PseudographGeneratorLLM:
    def __init__(self, model_name: str = "gemini-2.5-flash", key=None):
        # Activate dotenv

        load_dotenv()
        if key is None:
            key = os.getenv("PSEUDOGRAPH_RELABELLING_API_KEY")
        if not key:
            raise ValueError("API key missing.")

        self.client = genai.Client(api_key=key)
        self.model = model_name

        self.base_cfg = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
            system_instruction=[
                types.Part.from_text(text=FEWSHOT)
            ],
        )

    def generate(self, claim: str, max_tokens: int = 2048, retry: int = 3) -> str:
        """
        Return PURE triplets text (multi-line).
        Auto-retry on 503 or empty response.
        """
        prompt = f"Claim:\n\"{claim.strip()}\"\n\nTriplets:\n"

        # copy base config
        thinking_cfg = self.base_cfg.thinking_config

        for attempt in range(1, retry + 1):
            try:
                cfg = types.GenerateContentConfig(
                    temperature=self.base_cfg.temperature,
                    max_output_tokens=max_tokens,
                    thinking_config=thinking_cfg,
                    system_instruction=self.base_cfg.system_instruction,
                )

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=cfg
                )

                if not response or not response.text:
                    raise ValueError("Empty LLM response.")

                print("RES:", response.text.strip())

                out = response.text.strip()
                out = out.split("{")[0].strip()
                return out

            except ServerError as e:
                # handle server-side overload
                if "503" in str(e):
                    wait = 1.5 * attempt
                    print(f"[Retry {attempt}/{retry}] 503 Overloaded → waiting {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                else:
                    raise e

            except Exception as e:
                wait = 1.0 * attempt
                print(f"[Retry {attempt}/{retry}] Error: {e} → waiting {wait:.1f}s...")
                time.sleep(wait)
                continue

        # all retries failed
        raise RuntimeError(f"Failed after {retry} retries for claim: {claim}")


# if __name__ == "__main__":
#     generator = PseudographGeneratorLLM()
#     # claim = "The capital of France is not Berlin."
#     claim = "Hòa Bình Province is in the southwest of the city led by Ngô Thị Doãn Thanh, whose capital was established by the Nguyễn dynasty and is nicknamed 'Affluence'. It is northwest of Sơn La Province."
#     triplets = generator.generate(claim)
#     print("Generated Triplets:")
#     print(triplets)