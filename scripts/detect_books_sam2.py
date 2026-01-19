"""Detect books in an image with SAM 2 and highlight them with bounding boxes.

This script uses Meta's Segment Anything Model 2 (SAM 2) to propose masks in
an input image and then filters those proposals with a CLIP text/image model to
identify the ones that most resemble books. Bounding boxes for the selected
masks are drawn on top of the image, and the result can be saved to disk or
shown interactively.

Example
-------
python scripts/detect_books_sam2.py path/to/image.jpg \
    --output annotated.png --show

The script downloads weights for both SAM 2 and CLIP automatically via the
Hugging Face hub on first run. A GPU is recommended, but CPU execution works as
well (albeit more slowly).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import CLIPModel, CLIPProcessor


@dataclasses.dataclass
class Detection:
    """Simple structure describing a detected book mask."""

    bbox_xyxy: Tuple[int, int, int, int]
    area: float
    probability: float

    def as_patch(self, color: str = "lime") -> patches.Rectangle:
        x0, y0, x1, y1 = self.bbox_xyxy
        return patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the image to analyse.")
    parser.add_argument(
        "--model-id",
        default="facebook/sam2.1-hiera-small",
        help="Hugging Face model id for SAM 2 (default: %(default)s).",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face model id for CLIP (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device, e.g. 'cuda' or 'cpu'. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=200,
        help="Maximum number of SAM 2 masks to evaluate with CLIP (default: %(default)s).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=4_000,
        help="Ignore masks with area smaller than this many pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.35,
        help="Minimum CLIP probability for accepting a detection (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the annotated image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image in an interactive window.",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def load_sam2_generator(model_id: str, device: str) -> SAM2AutomaticMaskGenerator:
    print(f"Loading SAM 2 model '{model_id}' on {device} ...")
    return SAM2AutomaticMaskGenerator.from_pretrained(model_id, device=device)


def load_clip(model_id: str, device: str) -> Tuple[CLIPModel, CLIPProcessor]:
    print(f"Loading CLIP model '{model_id}' on {device} ...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def crop_masked_region(image: np.ndarray, mask: np.ndarray, bbox_xyxy: Sequence[float]) -> Image.Image:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox_xyxy]
    h, w = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        # Empty crop; return a black image for safety.
        return Image.new("RGB", (1, 1))

    crop = image[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    if mask_crop.ndim == 2:
        mask_crop = mask_crop[..., None]
    masked = crop.copy()
    masked[~mask_crop.astype(bool)] = 0
    return Image.fromarray(masked)


def batched(iterable: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def score_masks_with_clip(
    crops: Sequence[Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    text_prompts: Sequence[str],
) -> List[float]:
    probs: List[float] = []
    model = model.to(device)
    for batch in batched(crops, batch_size=8):
        inputs = processor(
            text=list(text_prompts),
            images=list(batch),
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits_per_image  # shape: (batch, len(text_prompts))
            batch_probs = logits.softmax(dim=-1)[:, 0]  # probability of the first prompt
        probs.extend(batch_probs.cpu().tolist())
    return probs


def run_detection(args: argparse.Namespace) -> List[Detection]:
    image = load_image(args.image)
    generator = load_sam2_generator(args.model_id, args.device)
    clip_model, clip_processor = load_clip(args.clip_model, args.device)

    print("Generating SAM 2 masks ...")
    proposals = generator.generate(image)
    print(f"Generated {len(proposals)} mask proposals.")

    filtered = [
        proposal
        for proposal in proposals
        if proposal["area"] >= args.min_area
    ]
    print(f"{len(filtered)} proposals remain after area filtering.")

    filtered.sort(key=lambda m: m.get("predicted_iou", 0.0), reverse=True)
    if args.max_masks and len(filtered) > args.max_masks:
        filtered = filtered[: args.max_masks]
        print(f"Keeping top {len(filtered)} proposals for CLIP scoring.")

    crops = [
        crop_masked_region(
            image,
            np.asarray(mask_info["segmentation"], dtype=bool),
            bbox_xyxy=_xywh_to_xyxy(mask_info["bbox"]),
        )
        for mask_info in filtered
    ]

    print("Scoring proposals with CLIP ...")
    text_prompts = ["a photo of a book", "a photo of something else"]
    probs = score_masks_with_clip(crops, clip_model, clip_processor, args.device, text_prompts)

    detections = []
    for mask_info, prob in zip(filtered, probs):
        if prob < args.min_prob:
            continue
        bbox_xyxy = _xywh_to_xyxy(mask_info["bbox"])
        detections.append(
            Detection(
                bbox_xyxy=tuple(map(int, bbox_xyxy)),
                area=float(mask_info["area"]),
                probability=float(prob),
            )
        )

    print(f"Detected {len(detections)} books above probability threshold {args.min_prob}.")
    annotate_and_maybe_save(image, detections, args)
    return detections


def annotate_and_maybe_save(image: np.ndarray, detections: Sequence[Detection], args: argparse.Namespace) -> None:
    if not detections and not args.output and not args.show:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    for det in detections:
        ax.add_patch(det.as_patch())
        x0, y0, _, _ = det.bbox_xyxy
        ax.text(
            x0,
            max(0, y0 - 5),
            f"book ({det.probability:.2f})",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1),
        )

    fig.tight_layout(pad=0)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, bbox_inches="tight", pad_inches=0)
        print(f"Saved annotated image to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def _xywh_to_xyxy(bbox_xywh: Sequence[float]) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))


if __name__ == "__main__":
    run_detection(parse_args())
