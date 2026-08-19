"""
pipeline.py
-----------
Layer 2 Orchestrator: Logic Parsing & NLP ("The Compiler")

Chains spaCy entity extraction -> Gemini gap-filling/structuring,
turning raw TrOCR token lines into the final JSON payload consumed by
the 3D shawl-rendering frontend.
"""

from dataclasses import dataclass, field
from typing import List

from .nlp_parser import TalimTokenParser, WeavingInstruction
from .gemini_corrector import GeminiSyntaxCorrector


@dataclass
class CompilationResult:
    instructions: List[WeavingInstruction] = field(default_factory=list)
    structured_json: dict = field(default_factory=dict)


class CompilerPipeline:
    """Layer 2: raw shorthand token lines -> render-ready JSON pattern."""

    def __init__(self, gemini_api_key: str, gemini_model: str = "gemini-2.5-pro"):
        self.parser = TalimTokenParser()
        self.corrector = GeminiSyntaxCorrector(api_key=gemini_api_key, model_name=gemini_model)

    def run(self, raw_token_lines: List[str]) -> CompilationResult:
        instructions = self.parser.parse_stream(raw_token_lines)
        structured = self.corrector.correct_and_structure(instructions)
        return CompilationResult(instructions=instructions, structured_json=structured)


if __name__ == "__main__":
    import os
    import json

    compiler = CompilerPipeline(gemini_api_key=os.environ["GEMINI_API_KEY"])
    result = compiler.run(["[RED_SYM] [4]", "[BLUE_SYM] [2]", "[torn]", "[RED_SYM] [4]"])
    print(json.dumps(result.structured_json, indent=2))