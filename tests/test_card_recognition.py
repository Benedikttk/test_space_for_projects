"""Tests for the ML card-recognition pipeline.

Covers:
* CardSynthesizer – image generation and dataset splitting
* CardYOLOTrainer – API surface (ultralytics not required to run these tests)
* TrainedCardYOLO – detector integration and build_detector factory

All tests run without a GPU and without downloading YOLO weights by using
monkeypatching where ultralytics would normally be needed.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_images(output_dir: Path, count: int = 10) -> list[Path]:
    """Write tiny (4x4) PNG files + YOLO labels to simulate generated data."""
    from PIL import Image
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(count):
        stem = f"A_S_{i+1:03d}"
        img  = Image.new("RGB", (4, 4), color=(200, 200, 200))
        p    = img_dir / f"{stem}.png"
        img.save(str(p))
        (lbl_dir / f"{stem}.txt").write_text(f"0 0.5 0.5 0.3 0.4\n")
        paths.append(p)
    return paths


# ===========================================================================
# PHASE 1: CardSynthesizer
# ===========================================================================

class TestCardSynthesizer:
    """Tests for blackjack.card_synthesizer.CardSynthesizer."""

    # ------------------------------------------------------------------
    # Module-level imports and constants
    # ------------------------------------------------------------------

    def test_all_cards_has_52_entries(self):
        from blackjack.card_synthesizer import ALL_CARDS
        assert len(ALL_CARDS) == 52

    def test_class_map_is_injective(self):
        from blackjack.card_synthesizer import CLASS_MAP
        values = list(CLASS_MAP.values())
        assert len(values) == len(set(values)), "Class IDs must be unique"

    def test_class_map_zero_indexed(self):
        from blackjack.card_synthesizer import CLASS_MAP
        assert min(CLASS_MAP.values()) == 0
        assert max(CLASS_MAP.values()) == 51

    # ------------------------------------------------------------------
    # Base card rendering
    # ------------------------------------------------------------------

    def test_render_base_card_returns_image(self):
        pytest.importorskip("PIL")
        from blackjack.card_synthesizer import _render_base_card, CARD_W, CARD_H
        img = _render_base_card("A", "S")
        assert img.size == (CARD_W, CARD_H)
        assert img.mode == "RGBA"

    @pytest.mark.parametrize("rank,suit", [
        ("A", "S"), ("T", "H"), ("2", "D"), ("K", "C"),
    ])
    def test_render_base_card_various_ranks(self, rank, suit):
        pytest.importorskip("PIL")
        from blackjack.card_synthesizer import _render_base_card
        img = _render_base_card(rank, suit)
        assert img is not None

    # ------------------------------------------------------------------
    # generate_card
    # ------------------------------------------------------------------

    def test_generate_card_creates_files(self, tmp_path):
        pytest.importorskip("PIL")
        from blackjack.card_synthesizer import CardSynthesizer
        gen = CardSynthesizer(output_dir=tmp_path)
        gen.generate_card("A", "S", variations=3)

        imgs = list((tmp_path / "images").glob("A_S_*.png"))
        lbls = list((tmp_path / "labels").glob("A_S_*.txt"))
        assert len(imgs) == 3
        assert len(lbls) == 3

    def test_generate_card_label_format(self, tmp_path):
        """Each label file must be: <class_id> <cx> <cy> <w> <h>."""
        pytest.importorskip("PIL")
        from blackjack.card_synthesizer import CardSynthesizer
        gen = CardSynthesizer(output_dir=tmp_path)
        gen.generate_card("2", "S", variations=1)

        lbl = next((tmp_path / "labels").glob("*.txt"))
        parts = lbl.read_text().strip().split()
        assert len(parts) == 5, "YOLO label must have 5 fields"
        class_id, cx, cy, bw, bh = [float(p) for p in parts]
        assert 0 <= class_id < 52
        assert 0.0 < cx < 1.0
        assert 0.0 < cy < 1.0
        assert 0.0 < bw <= 1.0
        assert 0.0 < bh <= 1.0

    def test_generate_card_unknown_card_raises(self, tmp_path):
        pytest.importorskip("PIL")
        from blackjack.card_synthesizer import CardSynthesizer
        gen = CardSynthesizer(output_dir=tmp_path)
        with pytest.raises(ValueError, match="Unknown card"):
            gen.generate_card("X", "Z", variations=1)

    # ------------------------------------------------------------------
    # generate_all_cards (smoke test with 1 variation)
    # ------------------------------------------------------------------

    def test_generate_all_cards_produces_52_types(self, tmp_path):
        pytest.importorskip("PIL")
        from blackjack.card_synthesizer import CardSynthesizer
        gen = CardSynthesizer(output_dir=tmp_path)
        gen.generate_all_cards(variations=1)

        imgs = list((tmp_path / "images").glob("*.png"))
        assert len(imgs) == 52

    # ------------------------------------------------------------------
    # create_yolo_dataset
    # ------------------------------------------------------------------

    def test_create_yolo_dataset_splits(self, tmp_path):
        pytest.importorskip("PIL")
        pytest.importorskip("yaml")
        from blackjack.card_synthesizer import CardSynthesizer
        synth_dir   = tmp_path / "synthetic"
        dataset_dir = tmp_path / "dataset"

        gen = CardSynthesizer(output_dir=synth_dir, canvas_size=(32, 32))
        gen.generate_all_cards(variations=1)           # 52 images total
        yaml_path = gen.create_yolo_dataset(dataset_dir=dataset_dir)

        assert yaml_path.exists()
        for split in ("train", "val", "test"):
            assert (dataset_dir / "images" / split).exists()
            assert (dataset_dir / "labels" / split).exists()

        n_train = len(list((dataset_dir / "images" / "train").glob("*.png")))
        n_val   = len(list((dataset_dir / "images" / "val").glob("*.png")))
        n_test  = len(list((dataset_dir / "images" / "test").glob("*.png")))
        assert n_train + n_val + n_test == 52

    def test_create_yolo_dataset_yaml_content(self, tmp_path):
        pytest.importorskip("PIL")
        import yaml
        from blackjack.card_synthesizer import CardSynthesizer
        synth_dir   = tmp_path / "synthetic"
        dataset_dir = tmp_path / "dataset"

        gen = CardSynthesizer(output_dir=synth_dir, canvas_size=(32, 32))
        gen.generate_all_cards(variations=1)
        yaml_path = gen.create_yolo_dataset(dataset_dir=dataset_dir)

        config = yaml.safe_load(yaml_path.read_text())
        assert config["nc"] == 52
        assert len(config["names"]) == 52
        assert "train" in config
        assert "val"   in config

    def test_create_yolo_dataset_no_images_raises(self, tmp_path):
        pytest.importorskip("PIL")
        from blackjack.card_synthesizer import CardSynthesizer
        gen = CardSynthesizer(output_dir=tmp_path / "empty")
        with pytest.raises(FileNotFoundError):
            gen.create_yolo_dataset(dataset_dir=tmp_path / "ds")

    def test_create_yolo_dataset_80_10_10_default_split(self, tmp_path):
        pytest.importorskip("PIL")
        import yaml
        from blackjack.card_synthesizer import CardSynthesizer

        synth_dir   = tmp_path / "synthetic"
        dataset_dir = tmp_path / "dataset"
        gen = CardSynthesizer(output_dir=synth_dir, canvas_size=(32, 32))
        gen.generate_all_cards(variations=1)
        gen.create_yolo_dataset(dataset_dir=dataset_dir)

        n_train = len(list((dataset_dir / "images" / "train").glob("*.png")))
        n_val   = len(list((dataset_dir / "images" / "val").glob("*.png")))
        # Allow off-by-one from floor division
        assert abs(n_train - int(52 * 0.80)) <= 1
        assert abs(n_val   - int(52 * 0.10)) <= 1


# ===========================================================================
# PHASE 2: CardYOLOTrainer
# ===========================================================================

class TestCardYOLOTrainer:
    """Tests for blackjack.card_yolo_trainer.CardYOLOTrainer."""

    def test_import_succeeds(self):
        from blackjack.card_yolo_trainer import CardYOLOTrainer  # noqa: F401

    def test_defaults(self):
        from blackjack.card_yolo_trainer import CardYOLOTrainer, _DEFAULT_DATA_YAML, _DEFAULT_MODEL_DIR
        t = CardYOLOTrainer()
        assert t.data_yaml == Path(_DEFAULT_DATA_YAML)
        assert t.model_dir == Path(_DEFAULT_MODEL_DIR)
        assert t.base_model == "yolov8n.pt"

    def test_train_raises_without_ultralytics(self, tmp_path):
        from blackjack.card_yolo_trainer import CardYOLOTrainer
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 52\nnames: []\ntrain: .\nval: .\n")
        trainer = CardYOLOTrainer(data_yaml=data_yaml, model_dir=tmp_path / "model")

        with patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises(ImportError, match="ultralytics"):
                trainer.train(epochs=1)

    def test_train_raises_when_yaml_missing(self, tmp_path):
        from blackjack.card_yolo_trainer import CardYOLOTrainer
        trainer = CardYOLOTrainer(
            data_yaml  = tmp_path / "missing.yaml",
            model_dir  = tmp_path / "model",
        )
        mock_yolo_module = MagicMock()
        with patch.dict("sys.modules", {"ultralytics": mock_yolo_module}):
            with pytest.raises(FileNotFoundError):
                trainer.train(epochs=1)

    def test_validate_raises_without_weights(self, tmp_path):
        from blackjack.card_yolo_trainer import CardYOLOTrainer
        trainer = CardYOLOTrainer(model_dir=tmp_path / "model")
        mock_yolo_module = MagicMock()
        with patch.dict("sys.modules", {"ultralytics": mock_yolo_module}):
            with pytest.raises(FileNotFoundError):
                trainer.validate()

    def test_test_on_real_images_returns_list(self, tmp_path):
        """test_on_real_images with a mocked model should return a list."""
        from PIL import Image
        from blackjack.card_yolo_trainer import CardYOLOTrainer

        # Create a dummy weight file so _get_model can load it
        weights_dir = tmp_path / "model" / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"fake")

        # Create a dummy test image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100)).save(str(img_path))

        mock_box = MagicMock()
        mock_box.cls   = [0]
        mock_box.conf  = [0.9]
        mock_box.xyxy  = [MagicMock(tolist=lambda: [10, 10, 50, 60])]

        mock_result = MagicMock()
        mock_result.names = {0: "A_S"}
        mock_result.boxes = [mock_box]

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.predict.return_value = [mock_result]

        mock_yolo_cls = MagicMock(return_value=mock_yolo_instance)
        mock_module   = MagicMock()
        mock_module.YOLO = mock_yolo_cls

        trainer = CardYOLOTrainer(model_dir=tmp_path / "model")
        with patch.dict("sys.modules", {"ultralytics": mock_module}):
            results = trainer.test_on_real_images(
                [str(img_path)], output_dir=tmp_path / "out"
            )

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["detections"][0]["rank"] == "A"
        assert results[0]["detections"][0]["suit"] == "S"


# ===========================================================================
# PHASE 3: TrainedCardYOLO + build_detector integration
# ===========================================================================

class TestTrainedCardYOLO:
    """Tests for the TrainedCardYOLO detector class."""

    def test_trained_card_yolo_unavailable_without_model(self, tmp_path):
        from blackjack.detector import TrainedCardYOLO
        d = TrainedCardYOLO(model_path=tmp_path / "nonexistent.pt")
        assert not d.is_available()

    def test_trained_card_yolo_detect_returns_empty_when_unavailable(self):
        from blackjack.detector import TrainedCardYOLO
        d = TrainedCardYOLO(model_path="does/not/exist.pt")
        result = d.detect(np.zeros((10, 10, 3), dtype=np.uint8))
        assert result == []

    def test_trained_card_yolo_detect_with_mock_model(self, tmp_path):
        from blackjack.detector import TrainedCardYOLO, DetectionResult

        mock_box = MagicMock()
        mock_box.cls  = [3]
        mock_box.conf = [0.88]
        mock_box.xyxy = [MagicMock(tolist=lambda: [5, 5, 55, 70])]

        mock_pred = MagicMock()
        mock_pred.names = {3: "K_H"}
        mock_pred.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_pred]

        d = TrainedCardYOLO.__new__(TrainedCardYOLO)
        d.model_path           = Path("fake.pt")
        d.confidence_threshold = 0.75
        d.device               = "cpu"
        d._model               = mock_model

        results = d.detect(np.zeros((100, 100, 3), dtype=np.uint8), source="dealer")
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, DetectionResult)
        assert r.rank   == "K"
        assert r.suit   == "H"
        assert r.source == "dealer"
        assert abs(r.confidence - 0.88) < 1e-6

    def test_trained_card_yolo_engine_rank_normalisation(self):
        from blackjack.detector import DetectionResult
        dr = DetectionResult(rank="K", suit="H", confidence=0.9)
        assert dr.engine_rank == "T"  # face cards → T in engine

    def test_build_detector_trained_yolo_falls_back_to_template(self, tmp_path):
        """build_detector('trained_yolo') with no weights falls back."""
        from blackjack.detector import build_detector, NullDetector, TemplateDetector

        # No weights file → TrainedCardYOLO.is_available() returns False
        # TemplateDetector also unavailable (empty dir) → NullDetector
        result = build_detector(
            backend="trained_yolo",
            model_path=str(tmp_path / "no_weights.pt"),
            template_dir=str(tmp_path / "no_templates"),
        )
        # Should fall back gracefully (NullDetector if no templates)
        assert result is not None

    def test_build_detector_trained_yolo_backend_string(self, tmp_path):
        """build_detector accepts 'trained_yolo' without raising."""
        from blackjack.detector import build_detector
        # Should not raise even with missing files
        detector = build_detector(backend="trained_yolo", model_path="nope.pt")
        assert detector is not None


# ===========================================================================
# PHASE 4: Accuracy helpers
# ===========================================================================

class TestAccuracyHelpers:
    """Lightweight accuracy / confusion matrix helpers."""

    def test_class_ids_cover_all_52_cards(self):
        from blackjack.card_synthesizer import CLASS_MAP, RANKS, SUITS
        for suit in SUITS:
            for rank in RANKS:
                assert (rank, suit) in CLASS_MAP

    def test_detections_parse_correctly_from_yolo_label(self, tmp_path):
        """The YOLO label produced by the synthesizer round-trips correctly."""
        pytest.importorskip("PIL")
        from blackjack.card_synthesizer import CardSynthesizer, CLASS_MAP

        gen = CardSynthesizer(output_dir=tmp_path, canvas_size=(32, 32))
        gen.generate_card("A", "S", variations=1)

        lbl_path = next((tmp_path / "labels").glob("*.txt"))
        parts = lbl_path.read_text().strip().split()
        class_id = int(parts[0])
        expected = CLASS_MAP[("A", "S")]
        assert class_id == expected
