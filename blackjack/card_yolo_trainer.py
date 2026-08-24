"""YOLO model trainer for playing-card detection.

Wraps Ultralytics YOLO training, validation, and inference so that the
rest of the blackjack system only has to call three simple methods.

Requires the ``vision_dl`` extra::

    pip install -e .[vision_dl]

Usage::

    from blackjack.card_yolo_trainer import CardYOLOTrainer

    trainer = CardYOLOTrainer()
    trainer.train(epochs=50, device="cpu")   # or device="0" for first GPU
    metrics = trainer.validate()
    trainer.test_on_real_images(["data/real_casino_images/test.jpg"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Default paths align with CardSynthesizer's output layout
_DEFAULT_DATA_YAML = "data/datasets/blackjack_cards/data.yaml"
_DEFAULT_MODEL_DIR = "data/models/card_detector"


class CardYOLOTrainer:
    """Train, validate, and test a YOLO v8 card-detection model.

    Parameters
    ----------
    data_yaml:
        Path to the ``data.yaml`` file produced by
        :meth:`CardSynthesizer.create_yolo_dataset`.
    model_dir:
        Directory where training results (weights + metrics) are saved.
    base_model:
        Ultralytics model checkpoint to fine-tune from.
        ``'yolov8n.pt'`` (nano) trains fast on CPU; use ``'yolov8s.pt'``
        or larger for higher accuracy if a GPU is available.
    """

    def __init__(
        self,
        data_yaml:   str | Path = _DEFAULT_DATA_YAML,
        model_dir:   str | Path = _DEFAULT_MODEL_DIR,
        base_model:  str        = "yolov8n.pt",
    ) -> None:
        self.data_yaml  = Path(data_yaml)
        self.model_dir  = Path(model_dir)
        self.base_model = base_model
        self._model: Optional[Any] = None   # ultralytics YOLO, loaded lazily

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        epochs:  int = 50,
        imgsz:   int = 416,
        device:  str = "cpu",
        patience: int = 10,
        batch:   int = 16,
    ) -> Path:
        """Fine-tune a YOLOv8 nano model on the synthetic card dataset.

        Parameters
        ----------
        epochs:
            Maximum training epochs (early stopping with *patience*).
        imgsz:
            Input image size (square).  416 is a good balance of speed
            and accuracy for card detection.
        device:
            ``'cpu'``, ``'0'`` (first GPU), ``'cuda'``, or ``'mps'``
            (Apple Silicon).
        patience:
            Number of epochs without improvement before early stopping.
        batch:
            Batch size.

        Returns
        -------
        Path to the best weights file (``best.pt``).
        """
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO training. "
                "Install with: pip install -e .[vision_dl]"
            ) from exc

        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"data.yaml not found at {self.data_yaml}. "
                "Run CardSynthesizer().create_yolo_dataset() first."
            )

        self.model_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(self.base_model)

        results = model.train(
            data      = str(self.data_yaml),
            epochs    = epochs,
            imgsz     = imgsz,
            device    = device,
            patience  = patience,
            batch     = batch,
            save      = True,
            project   = str(self.model_dir.parent),
            name      = self.model_dir.name,
            exist_ok  = True,
        )

        self._model = model

        best_pt = self.model_dir / "weights" / "best.pt"
        last_pt = self.model_dir / "weights" / "last.pt"

        # Save summary metrics next to the weights
        metrics_path = self.model_dir / "results.json"
        try:
            summary: Dict[str, Any] = {
                "epochs":    epochs,
                "imgsz":     imgsz,
                "device":    device,
                "best_pt":   str(best_pt),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall":    float(results.results_dict.get("metrics/recall(B)",    0)),
                "mAP50":     float(results.results_dict.get("metrics/mAP50(B)",     0)),
                "mAP50_95":  float(results.results_dict.get("metrics/mAP50-95(B)",  0)),
            }
            with open(metrics_path, "w") as fh:
                json.dump(summary, fh, indent=2)
            log.info("Training complete. Metrics: %s", summary)
        except Exception as exc:
            log.warning("Could not save metrics JSON: %s", exc)

        return best_pt

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, float]:
        """Run YOLO validation on the *val* split and return metrics.

        Returns
        -------
        Dictionary with keys: ``precision``, ``recall``, ``mAP50``,
        ``mAP50_95``.  Target values are P > 0.92, R > 0.90,
        mAP50 > 0.85.
        """
        model = self._get_model()
        results = model.val(data=str(self.data_yaml))
        metrics = {
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall":    float(results.results_dict.get("metrics/recall(B)",    0)),
            "mAP50":     float(results.results_dict.get("metrics/mAP50(B)",     0)),
            "mAP50_95":  float(results.results_dict.get("metrics/mAP50-95(B)",  0)),
        }
        log.info("Validation metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Inference on real images
    # ------------------------------------------------------------------

    def test_on_real_images(
        self,
        image_paths: List[str | Path],
        output_dir:  str | Path = "data/real_casino_images/results",
        conf_threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Run inference on a list of real casino photos.

        For each image the model draws bounding boxes + labels + confidence
        scores and saves the annotated result to *output_dir*.

        Parameters
        ----------
        image_paths:
            Paths to test images (JPEG, PNG, …).
        output_dir:
            Directory where annotated images are written.
        conf_threshold:
            Minimum confidence score to report a detection.

        Returns
        -------
        List of per-image result dicts, each containing:
        ``image``, ``detections`` (list of
        ``{rank, suit, confidence, bbox}``).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._get_model()
        all_results: List[Dict[str, Any]] = []

        for img_path in image_paths:
            img_path = Path(img_path)
            try:
                preds = model.predict(
                    str(img_path),
                    conf    = conf_threshold,
                    verbose = False,
                    save    = True,
                    project = str(output_dir),
                    name    = "inference",
                    exist_ok = True,
                )
            except Exception as exc:
                log.warning("Inference failed for %s: %s", img_path, exc)
                continue

            detections: List[Dict[str, Any]] = []
            for r in preds:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    label  = r.names[cls_id]
                    parts  = label.split("_", 1)
                    rank   = parts[0] if parts else label
                    suit   = parts[1] if len(parts) > 1 else ""
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "rank":       rank,
                        "suit":       suit,
                        "confidence": conf,
                        "bbox":       [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    })

            result = {"image": str(img_path), "detections": detections}
            all_results.append(result)
            log.info("%s → %d detection(s)", img_path.name, len(detections))

        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Return the loaded YOLO model, loading best.pt if needed."""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install with: pip install -e .[vision_dl]"
            ) from exc
        best_pt = self.model_dir / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(
                f"No trained weights at {best_pt}. "
                "Call train() first or point model_dir at an existing run."
            )
        self._model = YOLO(str(best_pt))
        return self._model
