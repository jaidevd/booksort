import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from ultralytics import YOLO

BOOK_CLASS_ID = 73  # COCO class index for books
DEFAULT_WEIGHTS = [
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
]
IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)
DEFAULT_CONF_SWEEP = [round(float(x), 2) for x in np.linspace(0.05, 0.95, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO11 segmentation models on test images, save highlighted detections, "
            "and compute mAP + inference timings."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("test_images"),
        help="Directory with evaluation images.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("labels"),
        help="Directory with polygon labels (JSON).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("book_detections"),
        help="Root directory where highlighted detections are written.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=DEFAULT_WEIGHTS,
        help="One or more YOLO11 segmentation weights to evaluate.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.1,
        help="Minimum confidence to keep a detection.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold applied to mask logits to obtain a binary mask.",
    )
    parser.add_argument(
        "--map-conf-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_CONF_SWEEP,
        help="Confidence thresholds used to compute the mAP curve.",
    )
    return parser.parse_args()


def load_ground_truth_masks(labels_dir: Path, images_dir: Path) -> Dict[str, List[np.ndarray]]:
    masks: Dict[str, List[np.ndarray]] = {}
    for label_path in sorted(labels_dir.glob("*.json")):
        image_name = label_path.stem
        image_path = find_image_path(images_dir, image_name)
        if image_path is None or not image_path.exists():
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        with open(label_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        polys = []
        for obj in annotations:
            points = [(pt["x"], pt["y"]) for pt in obj.get("content", [])]
            if len(points) >= 3:
                polys.append(points)
        masks[image_name] = [polygon_to_mask(poly, height, width) for poly in polys]
    return masks


def polygon_to_mask(points: Sequence[Sequence[float]], height: int, width: int) -> np.ndarray:
    mask_image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon(points, outline=1, fill=1)
    return np.array(mask_image, dtype=bool)


def find_image_path(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def upscale_masks(mask_tensor: torch.Tensor, orig_shape: Sequence[int]) -> np.ndarray:
    if mask_tensor.numel() == 0:
        return np.empty((0, *orig_shape), dtype=np.float32)
    resized = F.interpolate(
        mask_tensor.unsqueeze(1),
        size=orig_shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(1).cpu().numpy()


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_average_precision(predictions: List[dict], ground_truth: Dict[str, List[np.ndarray]]) -> float:
    gt_count = sum(len(masks) for masks in ground_truth.values())
    if gt_count == 0 or not predictions:
        return 0.0
    ap_scores = []
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    for threshold in IOU_THRESHOLDS:
        matched = {img_id: np.zeros(len(masks), dtype=bool) for img_id, masks in ground_truth.items()}
        tps: List[int] = []
        fps: List[int] = []
        for pred in predictions:
            image_id = pred["image_id"]
            pred_mask = pred["mask"]
            gt_masks = ground_truth.get(image_id, [])
            if image_id not in matched:
                matched[image_id] = np.zeros(len(gt_masks), dtype=bool)
            best_iou = 0.0
            best_idx = -1
            for idx, gt_mask in enumerate(gt_masks):
                if matched[image_id][idx]:
                    continue
                iou = mask_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= threshold and best_idx >= 0:
                matched[image_id][best_idx] = True
                tps.append(1)
                fps.append(0)
            else:
                tps.append(0)
                fps.append(1)
        if not tps:
            ap_scores.append(0.0)
            continue
        tps_cum = np.cumsum(tps)
        fps_cum = np.cumsum(fps)
        precisions = tps_cum / np.maximum(tps_cum + fps_cum, 1e-9)
        recalls = tps_cum / max(gt_count, 1)
        ap_scores.append(average_precision(recalls, precisions))
    return float(np.mean(ap_scores))


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def highlight_detection(base_image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    highlighted = base_image.copy()
    if not mask.any():
        return highlighted
    overlay = np.zeros_like(highlighted, dtype=np.float32)
    overlay[:, :] = np.array(color, dtype=np.float32)
    highlighted = highlighted.astype(np.float32)
    highlighted[mask] = (
        highlighted[mask] * (1 - alpha) + overlay[mask] * alpha
    )
    return highlighted.astype(np.uint8)


def predict_image(
    model: YOLO,
    image_path: Path,
    mask_threshold: float,
    record_threshold: float,
    highlight_threshold: float,
    highlight_root: Path,
) -> tuple[List[dict], float]:
    start = time.perf_counter()
    result = model(image_path)[0]
    inference_time = time.perf_counter() - start
    detections: List[dict] = []

    image_output_dir = highlight_root / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    if result.masks is None or result.boxes is None or not len(result.boxes):
        return detections, inference_time

    mask_tensor = result.masks.data.to(dtype=torch.float32, device="cpu")
    masks = upscale_masks(mask_tensor, result.masks.orig_shape)
    classes = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    base_image: Optional[np.ndarray] = None
    saved = 0
    for idx, (cls_id, score) in enumerate(zip(classes, scores)):
        if cls_id != BOOK_CLASS_ID or score < record_threshold:
            continue
        binary_mask = masks[idx] > mask_threshold
        if not binary_mask.any():
            continue
        detections.append(
            {
                "image_id": image_path.stem,
                "score": float(score),
                "mask": binary_mask,
            }
        )
        if score >= highlight_threshold:
            if base_image is None:
                base_image = np.array(Image.open(image_path).convert("RGB"))
            highlighted = highlight_detection(base_image, binary_mask)
            output_path = image_output_dir / f"{image_path.stem}_det{saved:02d}.png"
            Image.fromarray(highlighted).save(output_path)
            saved += 1
    return detections, inference_time


def evaluate_model(
    weight: str,
    image_paths: Iterable[Path],
    ground_truth: Dict[str, List[np.ndarray]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model = YOLO(weight)
    highlight_root = args.output_dir / Path(weight).stem
    highlight_root.mkdir(parents=True, exist_ok=True)
    detections: List[dict] = []
    inference_times: List[float] = []
    conf_thresholds = sorted(
        set(
            float(x)
            for x in (args.map_conf_thresholds + [args.conf_threshold])
            if 0.0 <= x <= 1.0
        )
    )
    if not conf_thresholds:
        raise ValueError("map_conf_thresholds must contain at least one value between 0 and 1.")
    record_threshold = conf_thresholds[0]
    for image_path in image_paths:
        preds, infer_time = predict_image(
            model,
            image_path,
            args.mask_threshold,
            record_threshold,
            args.conf_threshold,
            highlight_root,
        )
        detections.extend(preds)
        inference_times.append(infer_time)
    map_by_conf = {}
    for threshold in conf_thresholds:
        filtered = [pred for pred in detections if pred["score"] >= threshold]
        map_by_conf[threshold] = compute_average_precision(filtered, ground_truth)
    model_map = map_by_conf[args.conf_threshold]
    avg_infer = float(np.mean(inference_times)) if inference_times else 0.0
    return {
        "weight": weight,
        "mAP50-95": model_map,
        "avg_inference_time_sec": avg_infer,
        "total_detections": len(detections),
        "map_curve": map_by_conf,
        "conf_thresholds": conf_thresholds,
    }


def plot_map_vs_confidence(summary: List[dict], output_dir: Path) -> None:
    if not summary:
        return
    output_path = output_dir / "map_vs_confidence.png"
    plt.figure(figsize=(10, 6))
    for row in summary:
        thresholds = row["conf_thresholds"]
        curve = [row["map_curve"].get(th, 0.0) for th in thresholds]
        label = Path(row["weight"]).stem
        plt.plot(thresholds, curve, marker="o", label=label)
    plt.xlabel("Confidence threshold")
    plt.ylabel("mAP50-95 (IOU 0.5:0.95)")
    plt.title("mAP vs Confidence Threshold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved mAP vs confidence plot to {output_path}")


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        p for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")
    ground_truth = load_ground_truth_masks(args.labels_dir, args.images_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for weight in args.weights:
        if not weight:
            continue
        result = evaluate_model(weight, image_paths, ground_truth, args)
        summary.append(result)
        print(
            f"{result['weight']}: mAP50-95={result['mAP50-95']:.3f}, "
            f"avg inference={result['avg_inference_time_sec']:.3f}s, "
            f"detections={result['total_detections']}"
        )
    if summary:
        print("\n=== Summary ===")
        for row in summary:
            print(
                f"{row['weight']:>15} | mAP50-95: {row['mAP50-95']:.3f} | "
                f"avg inference: {row['avg_inference_time_sec']:.3f}s | "
                f"detections: {row['total_detections']}"
            )
        plot_map_vs_confidence(summary, args.output_dir)


if __name__ == "__main__":
    main()
