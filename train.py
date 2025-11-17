import argparse
import json
import random
import shutil
import textwrap
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune yolo11x-seg.pt on the polygon annotations stored in labels/. "
            "Images are expected in test_images/ by default."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("test_images"),
        help="Directory where raw JPG/PNG reference photos live.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("labels"),
        help="Directory with polygon annotations exported as JSON.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("booksort_yolo_dataset"),
        help="Destination for YOLO-formatted train/val splits.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolo11x-seg.pt"),
        help="Pretrained YOLO11 segmentation checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Image size used during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Mini-batch size used by YOLO.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Proportion of images reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic train/val splits.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs"),
        help="Root directory where YOLO training outputs are stored.",
    )
    parser.add_argument(
        "--run-name",
        default="booksort-finetune",
        help="Name for the YOLO training run.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience passed to YOLO.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string understood by YOLO (e.g. 0, 0,1, cpu).",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow YOLO to reuse an existing runs/<name> directory.",
    )
    parser.add_argument(
        "--overwrite-dataset",
        action="store_true",
        help="Wipe dataset-dir before writing YOLO-formatted files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_yaml = prepare_yolo_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        dataset_dir=args.dataset_dir,
        val_split=args.val_split,
        seed=args.seed,
        overwrite=args.overwrite_dataset,
    )
    model = YOLO(args.weights)
    train_kwargs = dict(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=str(args.project),
        name=args.run_name,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        exist_ok=args.exist_ok,
    )
    results = model.train(**train_kwargs)
    print("Training complete.")
    print(f"Results directory: {model.trainer.save_dir}")
    if hasattr(results, "save_dir"):
        print(f"Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")


def prepare_yolo_dataset(
    images_dir: Path,
    labels_dir: Path,
    dataset_dir: Path,
    val_split: float,
    seed: int,
    overwrite: bool = False,
) -> Path:
    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir {images_dir} does not exist.")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels-dir {labels_dir} does not exist.")
    if dataset_dir.exists():
        if overwrite:
            shutil.rmtree(dataset_dir)
        else:
            raise FileExistsError(
                f"{dataset_dir} already exists. Pass --overwrite-dataset to regenerate it."
            )
    samples = collect_samples(images_dir, labels_dir)
    if not samples:
        raise RuntimeError("No image/label pairs found. Nothing to train on.")
    random.seed(seed)
    random.shuffle(samples)
    val_count = int(round(len(samples) * val_split))
    if len(samples) > 1:
        val_count = min(max(val_count, 1), len(samples) - 1)
    else:
        val_count = 0
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]
    layout = create_dataset_tree(dataset_dir)
    for sample in train_samples:
        write_sample(sample, layout["train"])
    for sample in val_samples:
        write_sample(sample, layout["val"])
    yaml_path = dataset_dir / "booksort.yaml"
    yaml_path.write_text(render_dataset_yaml(dataset_dir), encoding="utf-8")
    print(
        f"Prepared {len(train_samples)} training files and {len(val_samples)} validation files "
        f"in {dataset_dir}."
    )
    return yaml_path


def collect_samples(images_dir: Path, labels_dir: Path) -> List[tuple[Path, Path]]:
    samples: List[tuple[Path, Path]] = []
    for label_path in sorted(labels_dir.glob("*.json")):
        image_path = find_image_path(images_dir, label_path.stem)
        if image_path is None:
            continue
        samples.append((image_path, label_path))
    return samples


def create_dataset_tree(dataset_dir: Path) -> dict:
    layout = {
        "train": {
            "images": dataset_dir / "images" / "train",
            "labels": dataset_dir / "labels" / "train",
        },
        "val": {
            "images": dataset_dir / "images" / "val",
            "labels": dataset_dir / "labels" / "val",
        },
    }
    for split in layout.values():
        split["images"].mkdir(parents=True, exist_ok=True)
        split["labels"].mkdir(parents=True, exist_ok=True)
    return layout


def write_sample(sample: tuple[Path, Path], split_paths: dict) -> None:
    image_path, label_path = sample
    target_image = split_paths["images"] / image_path.name
    target_label = split_paths["labels"] / f"{label_path.stem}.txt"
    shutil.copy2(image_path, target_image)
    segments = load_segments(label_path, image_path)
    write_label_file(target_label, segments)


def load_segments(label_path: Path, image_path: Path) -> List[List[float]]:
    with open(label_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    with Image.open(image_path) as img:
        width, height = img.size
    segments: List[List[float]] = []
    for obj in annotations:
        points = obj.get("content", [])
        if len(points) < 3:
            continue
        coords: List[float] = []
        for point in points:
            x = float(point["x"]) / width
            y = float(point["y"]) / height
            coords.extend([clamp01(x), clamp01(y)])
        if len(coords) >= 6:
            segments.append(coords)
    return segments


def write_label_file(label_path: Path, segments: Iterable[Sequence[float]]) -> None:
    lines: List[str] = []
    for segment in segments:
        coords = " ".join(f"{value:.6f}" for value in segment)
        lines.append(f"0 {coords}")
    label_text = "\n".join(lines)
    label_path.write_text(label_text, encoding="utf-8")


def render_dataset_yaml(dataset_dir: Path) -> str:
    resolved = dataset_dir.resolve()
    return textwrap.dedent(
        f"""\
        path: {resolved}
        train: images/train
        val: images/val
        names:
          0: book
        """
    )


def find_image_path(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    main()
