# BookSort

## Detect books in images with SAM 2

The repository now includes a convenience script that combines Meta's Segment Anything Model 2 (SAM 2) with CLIP to find books in a photograph and annotate them with bounding boxes.

### Usage

```bash
pip install sam2 transformers torch torchvision pillow matplotlib
python scripts/detect_books_sam2.py path/to/image.jpg --output annotated.png --show
```

Key options:

- `--model-id`: Hugging Face identifier for the SAM 2 checkpoint (default: `facebook/sam2.1-hiera-small`).
- `--clip-model`: CLIP checkpoint used to score each region (default: `openai/clip-vit-base-patch32`).
- `--min-prob`: Minimum probability assigned to the "book" prompt before a detection is kept (default: `0.35`).
- `--output`: Optional path for saving the annotated image. If omitted, results are only displayed when `--show` is provided.

The first run will download the SAM 2 and CLIP checkpoints automatically. A CUDA-capable GPU is recommended for best performance, but the script also works on CPU.
