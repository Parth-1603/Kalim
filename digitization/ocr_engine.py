"""
ocr_engine.py
-------------
Layer 1c: Sequence Recognition (Microsoft TrOCR)

Treats reading the Talim shorthand as image-to-text translation rather
than font-matching, since the symbols are custom and often connected.
A ViT encoder extracts visual features; a transformer decoder predicts
the token sequence, e.g. "[RED_SYM] [4] [BLUE_SYM] [2]".
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


@dataclass
class RecognizedLine:
    raw_tokens: str      # e.g. "[RED_SYM] [4] [BLUE_SYM] [2]"
    confidence: float


class TalimOCREngine:
    """Wraps a TrOCR model fine-tuned on Talim shorthand symbols."""

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _to_pil(self, crop: np.ndarray) -> Image.Image:
        # OpenCV crops are BGR; TrOCR's ViT backbone expects RGB.
        rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
        return Image.fromarray(rgb).convert("RGB")

    @torch.no_grad()
    def recognize_line(self, line_crop: np.ndarray) -> RecognizedLine:
        """Runs TrOCR on a single (already deskewed/oriented) line crop."""
        image = self._to_pil(line_crop)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        output = self.model.generate(
            pixel_values,
            max_length=64,
            num_beams=5,
            output_scores=True,
            return_dict_in_generate=True,
        )

        text = self.processor.batch_decode(output.sequences, skip_special_tokens=True)[0]

        # Approximate a line-level confidence from the beam search sequence score.
        seq_score = output.sequences_scores[0].item() if output.sequences_scores is not None else 0.0
        confidence = float(torch.exp(torch.tensor(seq_score)))

        return RecognizedLine(raw_tokens=text.strip(), confidence=confidence)

    def recognize_lines(self, line_crops: List[np.ndarray]) -> List[RecognizedLine]:
        """Batch-friendly wrapper preserving reading order."""
        return [self.recognize_line(crop) for crop in line_crops]


if __name__ == "__main__":
    import cv2

    engine = TalimOCREngine()
    sample_crop = cv2.imread("output/line_0.png")
    result = engine.recognize_line(sample_crop)
    print(f"Recognized: {result.raw_tokens} (confidence={result.confidence:.2f})")