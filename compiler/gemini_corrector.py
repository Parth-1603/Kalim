"""
gemini_corrector.py
--------------------
Layer 2b: Syntax Correction & Structuring (Google Gemini)

spaCy gives us structured-but-possibly-incomplete instructions (torn
paper, faded ink -> missing color or count). Gemini acts as the final
syntax checker: it looks at the *whole* pattern, uses the mathematical
symmetry that Kani weaving relies on to infer missing steps, and emits
a clean JSON payload for the 3D rendering frontend.
"""

import json
from dataclasses import asdict
from typing import List

import google.generativeai as genai

from .nlp_parser import WeavingInstruction


SYSTEM_PROMPT = """You are a syntax-correction engine for Kani shawl Talim scripts.
You will receive a JSON array of parsed weaving instructions, one per row,
each with: thread_color, warp_count, raw_line, is_complete.

Some rows may be incomplete because the source manuscript was torn,
faded, or otherwise illegible. Kani Talim patterns are mathematically
symmetric (repeating and mirrored motifs across rows). Use the
surrounding complete rows to infer the most likely thread_color and/or
warp_count for any incomplete row.

Return ONLY valid JSON (no markdown fences, no commentary) matching
this exact schema:

{
  "pattern": [
    {
      "row": <int>,
      "thread_color": "<string>",
      "warp_count": <int>,
      "inferred": <true|false>,
      "confidence": <float between 0 and 1>
    }
  ]
}
"""


class GeminiSyntaxCorrector:
    """Wraps a Gemini model call that fills gaps and structures final JSON."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
        )

    def _instructions_to_payload(self, instructions: List[WeavingInstruction]) -> str:
        rows = []
        for i, instr in enumerate(instructions):
            row = asdict(instr)
            row["row"] = i
            rows.append(row)
        return json.dumps(rows, ensure_ascii=False)

    def correct_and_structure(self, instructions: List[WeavingInstruction]) -> dict:
        """
        Sends parsed-but-possibly-gappy instructions to Gemini and returns
        the corrected, fully structured JSON payload ready for the 3D
        rendering frontend.
        """
        payload = self._instructions_to_payload(instructions)
        response = self.model.generate_content(
            payload,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # low temperature: this is inference, not creativity
                response_mime_type="application/json",
            ),
        )

        try:
            structured = json.loads(response.text)
        except (json.JSONDecodeError, AttributeError) as exc:
            raise ValueError(f"Gemini did not return valid JSON: {response}") from exc

        return structured


if __name__ == "__main__":
    import os
    from .nlp_parser import TalimTokenParser

    parser = TalimTokenParser()
    raw_lines = [
        "[RED_SYM] [4]",
        "[BLUE_SYM] [2]",
        "[torn]",  # illegible row -- Gemini should infer this from symmetry
        "[RED_SYM] [4]",
    ]
    parsed = parser.parse_stream(raw_lines)

    corrector = GeminiSyntaxCorrector(api_key=os.environ["GEMINI_API_KEY"])
    final_json = corrector.correct_and_structure(parsed)
    print(json.dumps(final_json, indent=2))