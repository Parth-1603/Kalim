"""
pipeline.py
-----------
Layer 1 Orchestrator: Digitization

Chains OpenCV preprocessing -> two-stage YOLO layout detection -> TrOCR
recognition, turning a raw manuscript photo into an ordered list of raw
shorthand token strings, one per detected line.
"""

from dataclasses import dataclass, field
from typing import List

from .preprocessing import TalimPreprocessor
from .layout_detection import TalimLayoutDetector
from .ocr_engine import TalimOCREngine, RecognizedLine


@dataclass
class DigitizationResult:
    source_path: str
    lines: List[RecognizedLine] = field(default_factory=list)

    @property
    def raw_token_stream(self) -> List[str]:
        """Flattened, reading-order list of raw token strings, one per line."""
        return [line.raw_tokens for line in self.lines]


class DigitizationPipeline:
    """Layer 1: turns a scanned Talim manuscript page into raw token sequences."""

    def __init__(
        self,
        preprocessor: TalimPreprocessor = None,
        layout_detector: TalimLayoutDetector = None,
        ocr_engine: TalimOCREngine = None,
    ):
        self.preprocessor = preprocessor or TalimPreprocessor()
        self.layout_detector = layout_detector or TalimLayoutDetector()
        self.ocr_engine = ocr_engine or TalimOCREngine()

    def run(self, image_path: str) -> DigitizationResult:
        clean_page = self.preprocessor.run(image_path)
        oriented_lines = self.layout_detector.run(clean_page)
        line_crops = [line.crop for line in oriented_lines]
        recognized = self.ocr_engine.recognize_lines(line_crops)
        return DigitizationResult(source_path=image_path, lines=recognized)


if __name__ == "__main__":
    pipeline = DigitizationPipeline()
    result = pipeline.run("samples/raw_manuscript_page.jpg")
    for i, tokens in enumerate(result.raw_token_stream):
        print(f"Line {i}: {tokens}")