import json
import os
from google import genai
from google.genai import types

class BasicSenseLLM:
    """
    A lightweight LLM to answer universal, obvious truths.
    If claim is meaningless → Unsupported
    If claim is obvious → Supported/Refuted
    Else → PassedDown (to KG-based reasoning)
    """

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv('BASIC_SENSE_LLM_API_KEY'))

    def submit(self, claim: str, max_tokens: int = 1024) -> dict:

        prompt = f"""
You are BasicSenseLLM — a filter LLM responsible for detecting whether a claim is:
1. A valid factual claim suitable for fact checking.
2. A universally obvious truth (e.g., Earth orbits the Sun, 1+1=2).
3. A claim that requires deeper KG reasoning.
4. Or meaningless / unsupported as a fact-checking query.

You MUST classify the input claim into exactly ONE of these categories:

### 1. Unsupported
Use this when the claim:
- is not a factual claim,
- is a personal statement (“Tôi là học sinh”, “I like pizza”),
- is a command (“Tính căn bậc 2 của 128”),
- is random text (“asdlfkja”, “abc xyz”),
- is ambiguous or not checkable as a fact.

Output:
{{
  "verdict": "Unsupported",
  "explanation": "Claim does not make sense"
}}

### 2. Supported / Refuted
Use only for obvious universal truths that nearly every human knows:
- “Earth orbits the Sun”
- “1+1=2”
- “Water boils at 100°C at sea level”
- “Cats are mammals”
→ Respond Supported / Refuted with a short explanation.

### 3. PassedDown
Use this when:
- the claim **is valid**, but
- **not an obvious universal truth**,
- requires knowledge about people, places, organizations, dates, events, etc.

Example:
“Barack Obama was born in Hawaii”
“Việt Nam có bao nhiêu tỉnh”
→ This must be PassedDown.

---

Your task:
- Evaluate the claim: "{claim}"
- Decide the correct verdict.
- Return STRICT JSON ONLY, no extra text.

Format:
{{
  "verdict": "Supported/Refuted/PassedDown/Unsupported",
  "explanation": "your explanation"
}}
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_tokens,
                system_instruction="""
You are BasicSenseLLM.
You NEVER produce chain-of-thought.
You produce concise answers.

Rules:
- If claim is not a factual statement → verdict=Unsupported.
- If it is a universally obvious fact → Supported/Refuted.
- If it is a valid claim requiring specific knowledge → PassedDown.
- If unsure, choose PassedDown.
- NEVER add ``` or markdown.
- Output must start with { and end with }.
- JSON must contain ONLY:
  - verdict
  - explanation
"""
            )
        )

        if not response or not response.text:
            raise ValueError("No response from BasicSenseLLM.")

        text = response.text.strip()

        try:
            parsed = json.loads(text)
            return {
                "verdict": parsed.get("verdict", ""),
                "explanation": parsed.get("explanation", "")
            }
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from BasicSenseLLM response: {text}")
