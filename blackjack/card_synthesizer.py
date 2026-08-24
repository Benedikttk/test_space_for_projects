"""Synthetic card image generator for YOLO training.

Generates photorealistic-looking playing card images with random
augmentations (rotation, perspective, brightness, contrast, blur,
JPEG artefacts, shadow, glare) and corresponding YOLO-format labels.

Usage::

    from blackjack.card_synthesizer import CardSynthesizer

    gen = CardSynthesizer()
    gen.generate_all_cards()   # ~6 000 images in data/synthetic_cards/
    gen.create_yolo_dataset()  # train/val/test split + data.yaml
"""

from __future__ import annotations

import io
import logging
import math
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Card type definitions
# ---------------------------------------------------------------------------

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['S', 'H', 'D', 'C']

SUIT_SYMBOL = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
SUIT_COLOR  = {'S': (20,  20,  20),  'H': (190,  20,  20),
               'D': (190,  20,  20), 'C': (20,  20,  20)}

# All 52 card types ordered consistently so class IDs are deterministic
ALL_CARDS: List[Tuple[str, str]] = [(r, s) for s in SUITS for r in RANKS]
CLASS_MAP: dict[Tuple[str, str], int] = {c: i for i, c in enumerate(ALL_CARDS)}

# Card dimensions (pixels) for the canonical base image
CARD_W, CARD_H = 160, 224


def _pil_available() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Base card renderer
# ---------------------------------------------------------------------------

def _render_base_card(rank: str, suit: str) -> "PIL.Image.Image":
    """Draw a clean playing card and return a PIL RGBA image."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (CARD_W, CARD_H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Card border
    draw.rounded_rectangle(
        [(2, 2), (CARD_W - 3, CARD_H - 3)],
        radius=12,
        outline=(180, 180, 180),
        width=2,
    )

    color = SUIT_COLOR[suit]
    sym   = SUIT_SYMBOL[suit]

    # Try to load a font; fall back gracefully
    try:
        font_rank  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_sym   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        font_big   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
    except Exception:
        font_rank = font_sym = font_big = ImageFont.load_default()

    # Top-left rank + suit
    draw.text((10, 8),  rank, fill=color, font=font_rank)
    draw.text((10, 38), sym,  fill=color, font=font_sym)

    # Bottom-right rank + suit (rotated 180°)
    tmp = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    td  = ImageDraw.Draw(tmp)
    td.text((10, 8),  rank, fill=color, font=font_rank)
    td.text((10, 38), sym,  fill=color, font=font_sym)
    rotated = tmp.rotate(180)
    img = Image.alpha_composite(img, rotated)
    draw = ImageDraw.Draw(img)

    # Centre large suit symbol
    draw.text((CARD_W // 2 - 18, CARD_H // 2 - 28), sym, fill=color, font=font_big)

    return img


# ---------------------------------------------------------------------------
# Augmentation helpers (pure-NumPy / PIL, no OpenCV required)
# ---------------------------------------------------------------------------

def _apply_rotation(img: "PIL.Image.Image", angle_deg: float) -> "PIL.Image.Image":
    from PIL import Image
    return img.rotate(angle_deg, expand=True, fillcolor=(200, 200, 200, 255))


def _apply_perspective(img: "PIL.Image.Image", strength: float = 0.08) -> "PIL.Image.Image":
    """Cheap 4-point perspective warp using affine approximation via PIL."""
    from PIL import Image
    w, h = img.size
    # Jitter the four corners slightly
    rng = random.random
    dx = int(w * strength * rng())
    dy = int(h * strength * rng())
    coeffs = _find_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],
        [(dx * rng(), dy * rng()),
         (w - dx * rng(), dy * rng()),
         (w - dx * rng(), h - dy * rng()),
         (dx * rng(), h - dy * rng())],
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def _find_coeffs(source_coords, target_coords):
    """Compute perspective transform coefficients for PIL.Image.transform."""
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.extend([
            [t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]],
            [0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]],
        ])
    A = np.array(matrix, dtype=float)
    B = np.array(source_coords, dtype=float).flatten()
    res, *_ = np.linalg.lstsq(A, B, rcond=None)
    return res.tolist()


def _apply_brightness_contrast(
    img: "PIL.Image.Image",
    brightness: float,
    contrast: float,
) -> "PIL.Image.Image":
    from PIL import ImageEnhance
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def _apply_blur(img: "PIL.Image.Image", radius: float) -> "PIL.Image.Image":
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _apply_jpeg_artifacts(img: "PIL.Image.Image", quality: int = 60) -> "PIL.Image.Image":
    """Encode/decode JPEG to simulate compression artefacts."""
    from PIL import Image
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy().convert("RGBA")


def _apply_shadow(img: "PIL.Image.Image") -> "PIL.Image.Image":
    """Overlay a semi-transparent gradient shadow on a random edge."""
    from PIL import Image
    w, h = img.size
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    arr = np.array(shadow)
    direction = random.choice(["top", "bottom", "left", "right"])
    alpha_max = int(random.uniform(40, 100))
    for i in range(max(w, h)):
        frac = i / max(w, h)
        a = int(alpha_max * (1 - frac))
        if direction == "top"    and i < h: arr[i, :, 3] = a
        elif direction == "bottom" and i < h: arr[h - i - 1, :, 3] = a
        elif direction == "left"   and i < w: arr[:, i, 3] = a
        elif direction == "right"  and i < w: arr[:, w - i - 1, 3] = a
    shadow = Image.fromarray(arr, "RGBA")
    return Image.alpha_composite(img, shadow)


def _apply_glare(img: "PIL.Image.Image") -> "PIL.Image.Image":
    """Overlay a bright elliptical spot (camera flash / glare)."""
    from PIL import Image
    w, h = img.size
    glare = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    arr = np.array(glare, dtype=np.float32)
    cx, cy = random.uniform(0.2, 0.8) * w, random.uniform(0.2, 0.8) * h
    rx, ry = random.uniform(0.1, 0.3) * w, random.uniform(0.1, 0.3) * h
    intensity = random.uniform(80, 160)
    ys, xs = np.mgrid[0:h, 0:w]
    dist = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
    mask = np.clip(1 - dist, 0, 1)
    arr[:, :, 0] = 255
    arr[:, :, 1] = 255
    arr[:, :, 2] = 255
    arr[:, :, 3] = (mask * intensity).clip(0, 200)
    glare = Image.fromarray(arr.astype(np.uint8), "RGBA")
    return Image.alpha_composite(img, glare)


# ---------------------------------------------------------------------------
# Main synthesizer class
# ---------------------------------------------------------------------------

class CardSynthesizer:
    """Generate synthetic card images for YOLO training.

    Parameters
    ----------
    output_dir:
        Root directory for generated images and labels.
    canvas_size:
        Size of the output image canvas in pixels (width, height).
        The card is placed on a coloured background of this size.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/synthetic_cards",
        canvas_size: Tuple[int, int] = (320, 320),
    ) -> None:
        self.output_dir  = Path(output_dir)
        self.canvas_size = canvas_size
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_card(self, rank: str, suit: str, variations: int = 100) -> None:
        """Generate *variations* augmented images for one card type.

        Each image is saved as ``<rank>_<suit>_NNN.png`` together with a
        YOLO-format ``.txt`` annotation file in the ``labels/`` sibling
        directory.

        Parameters
        ----------
        rank:
            Card rank: '2'-'9', 'T', 'J', 'Q', 'K', 'A'.
        suit:
            Card suit: 'S', 'H', 'D', 'C'.
        variations:
            Number of augmented images to produce (default 100).
        """
        if not _pil_available():
            raise ImportError("Pillow is required for card synthesis. "
                              "Install it with: pip install Pillow")

        class_id = CLASS_MAP.get((rank, suit))
        if class_id is None:
            raise ValueError(f"Unknown card: {rank}_{suit}")

        base = _render_base_card(rank, suit)

        for i in range(1, variations + 1):
            canvas, bbox_norm = self._augment_and_place(base)
            stem = f"{rank}_{suit}_{i:03d}"
            canvas.convert("RGB").save(
                self.output_dir / "images" / f"{stem}.png"
            )
            self._write_yolo_label(
                self.output_dir / "labels" / f"{stem}.txt",
                class_id,
                bbox_norm,
            )

        log.debug("Generated %d images for %s_%s", variations, rank, suit)

    def generate_all_cards(self, variations: int = 100) -> None:
        """Generate *variations* images for all 52 card types.

        Total output: ``52 × variations`` images (default 5 200).
        """
        total = len(ALL_CARDS)
        for idx, (rank, suit) in enumerate(ALL_CARDS, 1):
            log.info("[%d/%d] Generating %s_%s …", idx, total, rank, suit)
            self.generate_card(rank, suit, variations=variations)
        log.info(
            "Done. %d images written to %s",
            total * variations,
            self.output_dir / "images",
        )

    def create_yolo_dataset(
        self,
        dataset_dir: str | Path = "data/datasets/blackjack_cards",
        train_frac: float = 0.80,
        val_frac:   float = 0.10,
    ) -> Path:
        """Organise generated images into train/val/test splits.

        Creates the directory tree expected by Ultralytics YOLO and writes
        ``data.yaml``.

        Parameters
        ----------
        dataset_dir:
            Root of the YOLO dataset.
        train_frac:
            Fraction of images used for training (default 0.80).
        val_frac:
            Fraction used for validation (default 0.10).
            The remaining images go to *test*.

        Returns
        -------
        Path to the generated ``data.yaml`` file.
        """
        import yaml  # type: ignore

        dataset_dir = Path(dataset_dir)
        for split in ("train", "val", "test"):
            (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        images = sorted((self.output_dir / "images").glob("*.png"))
        if not images:
            raise FileNotFoundError(
                f"No images found in {self.output_dir / 'images'}. "
                "Run generate_all_cards() first."
            )

        rng = random.Random(42)
        shuffled = images[:]
        rng.shuffle(shuffled)

        n       = len(shuffled)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        splits = {
            "train": shuffled[:n_train],
            "val":   shuffled[n_train: n_train + n_val],
            "test":  shuffled[n_train + n_val:],
        }

        for split, img_paths in splits.items():
            for img_path in img_paths:
                lbl_path = self.output_dir / "labels" / (img_path.stem + ".txt")
                shutil.copy2(img_path,  dataset_dir / "images" / split / img_path.name)
                if lbl_path.exists():
                    shutil.copy2(lbl_path, dataset_dir / "labels" / split / lbl_path.name)

        # Write data.yaml
        yaml_path = dataset_dir / "data.yaml"
        names = [f"{r}_{s}" for s in SUITS for r in RANKS]
        config = {
            "path":  str(dataset_dir.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "nc":    len(names),
            "names": names,
        }
        with open(yaml_path, "w") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

        log.info(
            "Dataset created at %s  (train=%d, val=%d, test=%d)",
            dataset_dir,
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )
        return yaml_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _augment_and_place(
        self, base: "PIL.Image.Image"
    ) -> Tuple["PIL.Image.Image", Tuple[float, float, float, float]]:
        """Apply random augmentations to *base* and place on a canvas.

        Returns (canvas_image, (cx, cy, w, h)) where all coordinates are
        normalised to [0, 1] relative to the canvas.
        """
        from PIL import Image

        img = base.copy()
        cw, ch = self.canvas_size

        # --- Augmentations ---
        angle = random.uniform(-45, 45)
        img = _apply_rotation(img, angle)

        if random.random() < 0.6:
            img = _apply_perspective(img, strength=random.uniform(0.03, 0.10))

        brightness = random.uniform(0.7, 1.3)
        contrast   = random.uniform(0.8, 1.2)
        img = _apply_brightness_contrast(img, brightness, contrast)

        if random.random() < 0.5:
            img = _apply_blur(img, radius=random.uniform(0.5, 2.0))

        if random.random() < 0.4:
            img = _apply_jpeg_artifacts(img, quality=random.randint(50, 85))

        if random.random() < 0.5:
            img = _apply_shadow(img)

        if random.random() < 0.3:
            img = _apply_glare(img)

        # --- Scale so the card fits comfortably on the canvas ---
        max_scale = min(cw / img.width, ch / img.height) * random.uniform(0.55, 0.85)
        new_w = max(1, int(img.width  * max_scale))
        new_h = max(1, int(img.height * max_scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Random placement, keeping card fully inside canvas
        x0 = random.randint(0, max(0, cw - new_w))
        y0 = random.randint(0, max(0, ch - new_h))

        # Build canvas with random background colour
        bg_color = tuple(random.randint(30, 90) for _ in range(3))
        canvas = Image.new("RGBA", (cw, ch), bg_color + (255,))
        canvas.paste(img, (x0, y0), img)

        # YOLO bbox (normalised cx, cy, w, h)
        cx = (x0 + new_w / 2) / cw
        cy = (y0 + new_h / 2) / ch
        bw = new_w / cw
        bh = new_h / ch
        return canvas, (cx, cy, bw, bh)

    @staticmethod
    def _write_yolo_label(
        path: Path,
        class_id: int,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        cx, cy, bw, bh = bbox
        with open(path, "w") as fh:
            fh.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
