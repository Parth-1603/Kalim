"""
layout_detection.py
--------------------
Layer 1b: Document Layout & Oriented Bounding Box Detection (YOLO)

Two-stage detector:
  Stage 1 - DocLayout-YOLO: macro-level page segmentation. Separates the
            core Talim script region from margins, borders, and damage.
  Stage 2 - YOLO-OBB (YOLO26-obb): micro-level line/symbol localization
            using rotated boxes, since shorthand lines are rarely
            perfectly horizontal on aged paper.
"""

from dataclasses import dataclass
from typing import List
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class LayoutRegion:
    label: str          # e.g. "talim_text_block", "margin", "damage"
    bbox: tuple          # (x1, y1, x2, y2) axis-aligned
    confidence: float


@dataclass
class OrientedLine:
    points: np.ndarray   # 4x2 array of rotated box corners
    confidence: float
    crop: np.ndarray     # perspective-corrected crop ready for TrOCR


class TalimLayoutDetector:
    """Two-stage YOLO pipeline for isolating handwritten shorthand lines."""

    # Class of interest from the macro-segmentation model
    SCRIPT_LABEL = "talim_text_block"

    def __init__(
        self,
        doclayout_weights: str = "models/doclayout_yolo.pt",
        obb_weights: str = "models/yolo26_obb_talim.pt",
        conf_threshold: float = 0.35,
    ):
        self.page_model = YOLO(doclayout_weights)
        self.obb_model = YOLO(obb_weights)
        self.conf_threshold = conf_threshold

    # ---------- Stage 1: macro page segmentation ----------
    def segment_page(self, image: np.ndarray) -> List[LayoutRegion]:
        """Runs DocLayout-YOLO to find the Talim script block(s) on the page."""
        results = self.page_model.predict(image, conf=self.conf_threshold, verbose=False)
        regions = []
        for r in results:
            for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                label = r.names[int(cls_id)]
                regions.append(
                    LayoutRegion(
                        label=label,
                        bbox=tuple(box.cpu().numpy().astype(int)),
                        confidence=float(conf),
                    )
                )
        return regions

    def get_script_crops(self, image: np.ndarray) -> List[np.ndarray]:
        """Filters segmented regions down to only the Talim script blocks."""
        regions = self.segment_page(image)
        crops = []
        for region in regions:
            if region.label == self.SCRIPT_LABEL:
                x1, y1, x2, y2 = region.bbox
                crops.append(image[y1:y2, x1:x2])
        return crops

    # ---------- Stage 2: micro oriented-box line/symbol detection ----------
    def detect_oriented_lines(self, script_crop: np.ndarray) -> List[OrientedLine]:
        """
        Runs YOLO-OBB on a script block to find individual shorthand lines,
        each potentially rotated, and returns perspective-corrected crops.
        """
        results = self.obb_model.predict(script_crop, conf=self.conf_threshold, verbose=False)
        lines = []
        for r in results:
            if r.obb is None:
                continue
            for poly, conf in zip(r.obb.xyxyxyxy, r.obb.conf):
                pts = poly.cpu().numpy().reshape(4, 2).astype(np.float32)
                warped = self._warp_to_rectangle(script_crop, pts)
                lines.append(OrientedLine(points=pts, confidence=float(conf), crop=warped))
        # Sort top-to-bottom, matching natural reading order of a Talim script
        lines.sort(key=lambda ln: ln.points[:, 1].min())
        return lines

    @staticmethod
    def _warp_to_rectangle(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Un-rotates an oriented bounding box into a clean horizontal crop."""
        width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        height = int(max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[3] - pts[0])))
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(image, matrix, (width, height))

    def run(self, preprocessed_page: np.ndarray) -> List[OrientedLine]:
        """Full layer-1b pipeline: page -> script blocks -> oriented line crops."""
        all_lines: List[OrientedLine] = []
        for script_block in self.get_script_crops(preprocessed_page):
            all_lines.extend(self.detect_oriented_lines(script_block))
        return all_lines


if __name__ == "__main__":
    page = cv2.imread("output/clean_page.png")
    detector = TalimLayoutDetector()
    lines = detector.run(page)
    print(f"Detected {len(lines)} shorthand lines ready for TrOCR.")