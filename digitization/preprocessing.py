"""
preprocessing.py
-----------------
Layer 1a: Image Preprocessing (OpenCV)

Cleans raw scans of historical Talim manuscripts -- faded ink, paper
stains, uneven lighting -- so downstream YOLO / TrOCR models get a
high-contrast, noise-free image to work with.
"""

import cv2
import numpy as np
from pathlib import Path


class TalimPreprocessor:
    """Prepares a raw scanned Talim manuscript page for layout detection and OCR."""

    def __init__(self, blur_kernel: tuple = (5, 5), block_size: int = 35, c: int = 11):
        """
        Args:
            blur_kernel: Kernel size for Gaussian blur (must be odd numbers).
            block_size: Neighborhood size used by adaptive thresholding (must be odd, >1).
            c: Constant subtracted from the mean in adaptive thresholding.
        """
        self.blur_kernel = blur_kernel
        self.block_size = block_size
        self.c = c

    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        return img

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Gaussian blur to smooth paper-grain noise and stains before thresholding."""
        return cv2.GaussianBlur(gray, self.blur_kernel, 0)

    def adaptive_binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding handles uneven lighting/fading across an old
        manuscript far better than a single global threshold would.
        """
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.c,
        )

    def morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Opening removes small speckle noise (ink dots, dust);
        closing reconnects thin strokes broken by paper degradation.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closed

    def deskew(self, binary: np.ndarray) -> np.ndarray:
        """Corrects slight page rotation from imperfect scanning."""
        coords = np.column_stack(np.where(binary > 0))
        if coords.size == 0:
            return binary
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

        (h, w) = binary.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            binary, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    def run(self, path: str, save_path: str = None) -> np.ndarray:
        """Full preprocessing pipeline: raw scan -> clean binary image."""
        img = self.load_image(path)
        gray = self.to_grayscale(img)
        blurred = self.denoise(gray)
        binary = self.adaptive_binarize(blurred)
        cleaned = self.morphological_cleanup(binary)
        deskewed = self.deskew(cleaned)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, deskewed)

        return deskewed


if __name__ == "__main__":
    pre = TalimPreprocessor()
    clean_img = pre.run("samples/raw_manuscript_page.jpg", "output/clean_page.png")
    print("Preprocessing complete. Clean image shape:", clean_img.shape)