"""
main.py
-------
End-to-end Kani Talim compiler.

    Manuscript photo
        -> [Layer 1: Digitization]  OpenCV -> YOLO (DocLayout + OBB) -> TrOCR
        -> [Layer 2: Compiler]      spaCy NER -> Gemini correction/structuring
        -> render-ready JSON for the 3D shawl-rendering frontend
"""

import argparse
import json
import os
from pathlib import Path

from digitization.pipeline import DigitizationPipeline
from compiler.pipeline import CompilerPipeline


def compile_manuscript(image_path: str, gemini_api_key: str, out_path: str = None) -> dict:
    print(f"[1/2] Digitizing manuscript: {image_path}")
    digitizer = DigitizationPipeline()
    digitization_result = digitizer.run(image_path)

    print(f"      -> {len(digitization_result.lines)} shorthand lines extracted")
    for i, line in enumerate(digitization_result.lines):
        print(f"         line {i}: {line.raw_tokens!r} (conf={line.confidence:.2f})")

    print("[2/2] Compiling to structured pattern (spaCy -> Gemini)")
    compiler = CompilerPipeline(gemini_api_key=gemini_api_key)
    compilation_result = compiler.run(digitization_result.raw_token_stream)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(compilation_result.structured_json, f, indent=2, ensure_ascii=False)
        print(f"Saved render-ready JSON -> {out_path}")

    return compilation_result.structured_json


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Digitize and compile a Kani Talim manuscript page.")
    ap.add_argument("image", help="Path to the scanned manuscript image")
    ap.add_argument("--out", default="output/pattern.json", help="Where to save the structured JSON")
    ap.add_argument(
        "--gemini-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key (defaults to GEMINI_API_KEY env var)",
    )
    args = ap.parse_args()

    if not args.gemini_key:
        raise SystemExit("Set GEMINI_API_KEY or pass --gemini-key")

    result = compile_manuscript(args.image, args.gemini_key, args.out)
    print(json.dumps(result, indent=2))