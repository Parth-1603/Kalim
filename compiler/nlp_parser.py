"""
nlp_parser.py
-------------
Layer 2a: Named Entity Recognition (spaCy)

Takes the raw token stream produced by TrOCR (e.g. "[RED_SYM] [4]
[BLUE_SYM] [2]") and maps each token to a structured weaving concept:
Thread Color, Warp Count, Row Instruction, etc. This is what turns
"text on a page" into "instructions a loom can act on."
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy.matcher import Matcher


@dataclass
class WeavingInstruction:
    thread_color: Optional[str] = None
    warp_count: Optional[int] = None
    raw_line: str = ""
    entities: List[dict] = field(default_factory=list)  # [{"text":..,"label":..}]
    is_complete: bool = False  # False if color or count is missing (torn manuscript)


class TalimTokenParser:
    """
    Custom spaCy pipeline over the bracketed shorthand token stream.

    Rather than tokenizing the raw string with spaCy's default English
    tokenizer, we register a custom tokenizer that treats each
    "[TOKEN]" as a single spaCy token, then attach a rule-based Matcher
    to tag Thread Color / Warp Count entities.
    """

    TOKEN_PATTERN = re.compile(r"\[[^\]]+\]")

    # Known shorthand color symbols mapped to human-readable thread colors.
    COLOR_MAP = {
        "RED_SYM": "red",
        "BLUE_SYM": "blue",
        "GOLD_SYM": "gold",
        "WHITE_SYM": "white",
        "BLACK_SYM": "black",
        "GREEN_SYM": "green",
    }

    def __init__(self, model: str = "en_core_web_sm"):
        # We reuse spaCy's pipeline machinery but supply a custom tokenizer
        # since Talim shorthand isn't natural-language English text.
        self.nlp: Language = spacy.blank("en")
        self.nlp.tokenizer = self._build_custom_tokenizer(self.nlp)
        self.matcher = Matcher(self.nlp.vocab)
        self._register_patterns()

    def _build_custom_tokenizer(self, nlp: Language):
        from spacy.tokenizer import Tokenizer

        def tokenizer(text: str) -> Doc:
            tokens = self.TOKEN_PATTERN.findall(text) or text.split()
            return Doc(nlp.vocab, words=tokens, spaces=[True] * len(tokens))

        return tokenizer

    def _register_patterns(self):
        color_syms = [[{"TEXT": f"[{sym}]"}] for sym in self.COLOR_MAP]
        for pattern in color_syms:
            self.matcher.add("THREAD_COLOR", [pattern])

        # A bracketed integer following a color/other symbol is a warp count.
        self.matcher.add("WARP_COUNT", [[{"TEXT": {"REGEX": r"^\[\d+\]$"}}]])

    def parse_line(self, raw_line: str) -> WeavingInstruction:
        """Parses a single raw TrOCR token line into a structured instruction."""
        doc = self.nlp(raw_line)
        matches = self.matcher(doc)

        instruction = WeavingInstruction(raw_line=raw_line)

        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span: Span = doc[start:end]
            instruction.entities.append({"text": span.text, "label": label})

            if label == "THREAD_COLOR":
                sym = span.text.strip("[]")
                instruction.thread_color = self.COLOR_MAP.get(sym)
            elif label == "WARP_COUNT":
                instruction.warp_count = int(span.text.strip("[]"))

        instruction.is_complete = (
            instruction.thread_color is not None and instruction.warp_count is not None
        )
        return instruction

    def parse_stream(self, raw_lines: List[str]) -> List[WeavingInstruction]:
        return [self.parse_line(line) for line in raw_lines]


if __name__ == "__main__":
    parser = TalimTokenParser()
    sample = ["[RED_SYM] [4] [BLUE_SYM] [2]", "[GOLD_SYM] [torn]"]
    for instr in parser.parse_stream(sample):
        print(instr)