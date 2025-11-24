import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List
import os
import math
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import BisectingKMeans
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from matplotlib.path import Path as MplPath
from tqdm import tqdm
from ultralytics import YOLO

BOOK_CLASS_ID = 73  # COCO id for "book"
MODEL = YOLO("yolo11x-seg.pt", task="segment")
CACHE_DIR = Path("ocr_cache")
BOOKS_CACHE_DIR = Path("books_cache")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
op = os.path


def gather_image_paths(target: str) -> List[str]:
    """Return a sorted list of image paths from either a single file or directory."""
    path = Path(target)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return [
            str(p)
            for p in sorted(path.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
    raise FileNotFoundError(f"{target} is neither a file nor a directory.")


def md5(path: str) -> str:
    checksum = hashlib.md5()
    with open(path, "rb") as fin:
        checksum.update(fin.read())
    return checksum.hexdigest()


def ensure_client() -> tuple[DocumentIntelligenceClient | None, str]:
    """Load Azure and Google API clients if .secrets.json is configured."""
    if not Path(".secrets.json").exists():
        return None, ""
    with open(".secrets.json", "r", encoding="utf-8") as fin:
        secrets = json.load(fin)
    credential = AzureKeyCredential(secrets["AZURE_API_KEY"])
    di_client = DocumentIntelligenceClient(secrets["AZURE_DI_ENDPOINT"], credential)
    return di_client, secrets["GOOGLE_BOOKS_API_KEY"]


def detect_books(img_path: str) -> List[Dict]:
    """Run YOLO and return book detections with their polygons and min-area rects."""
    results = MODEL(img_path)[0]
    if results.masks is None:
        return []

    detections = []
    for cls_id, conf, polygon in zip(
        results.boxes.cls, results.boxes.conf, results.masks.xy
    ):
        if int(cls_id.item()) != BOOK_CLASS_ID:
            continue
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        rect = min_area_rect(pts)
        detections.append(
            {"polygon": pts, "min_rect": rect, "confidence": float(conf.item())}
        )
    return detections


def min_area_rect(points: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(points[:, None, :])
    box = cv2.boxPoints(rect)
    return box


def longest_edge(rect: np.ndarray) -> float:
    """Return the longest edge length for a 4-point rectangle."""
    diffs = rect - np.roll(rect, -1, axis=0)
    return float(np.max(np.linalg.norm(diffs, axis=1)))


def cap_image_size_once(path, quality=85, maxsize=4 * 1024 * 1024):
    original_size = op.getsize(path)
    if original_size <= maxsize:
        return

    scale = math.sqrt(maxsize / original_size)

    img = Image.open(path)
    w = max(1, int(img.width * scale))
    h = max(1, int(img.height * scale))

    img.resize((w, h), Image.LANCZOS).save(path, format=img.format,
                                           quality=quality, optimize=True)


def ocr(img_path: str, di_client: DocumentIntelligenceClient | None) -> List[Dict]:
    """Return cached OCR paragraphs with their polygons and centroids."""
    CACHE_DIR.mkdir(exist_ok=True)
    checksum = md5(img_path)
    cache_path = CACHE_DIR / f"{checksum}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        result = AnalyzeResult(data)
    else:
        if di_client is None:
            raise RuntimeError(
                "OCR cache miss. Provide .secrets.json with Azure credentials."
            )
        cap_image_size_once(img_path)
        with open(img_path, "rb") as fin:
            poller = di_client.begin_analyze_document("prebuilt-read", body=fin)
        result = poller.result()
        with open(cache_path, "w", encoding="utf-8") as fout:
            json.dump(result.as_dict(), fout, indent=2)

    entries = []
    for para in getattr(result, "paragraphs", []):
        if not para.bounding_regions:
            continue
        text = para.content.strip()
        if not text:
            continue
        polygon = np.array(para.bounding_regions[0].polygon, dtype=np.float32).reshape(
            -1, 2
        )
        centroid = polygon.mean(axis=0)
        entries.append({"text": text, "polygon": polygon, "centroid": centroid})
    return entries


def search(query: str, key: str) -> Dict:
    """Query Google Books for `query` and return the first hit's metadata."""
    if not key:
        return {}
    BOOKS_CACHE_DIR.mkdir(exist_ok=True)
    cache_key = hashlib.md5(query.encode("utf-8")).hexdigest()
    cache_path = BOOKS_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as fin:
            return json.load(fin)
    try:
        response = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={
                "q": query,
                "orderBy": "relevance",
                "maxResults": 5,
                "projection": "full",
                "key": key,
            },
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Google Books search failed for '{query[:40]}': {exc}")
        with open(cache_path, "w", encoding="utf-8") as fout:
            json.dump({"error": str(exc)}, fout, indent=2)
        return {}

    result = response.json()
    item = result.get("items", [False])[0]
    if not item:
        with open(cache_path, "w", encoding="utf-8") as fout:
            json.dump({}, fout, indent=2)
        return {}

    record = {"id": item["id"]}
    vinfo = item["volumeInfo"]
    record["title"] = vinfo.get("title", "")
    record["authors"] = vinfo.get("authors", [])
    record["isbn"] = vinfo.get("industryIdentifiers", [])
    record["categories"] = vinfo.get("categories", [])

    try:
        detail_resp = requests.get(
            f'https://www.googleapis.com/books/v1/volumes/{item["id"]}',
            params={"projection": "full", "key": key},
            timeout=10,
        )
        detail_resp.raise_for_status()
        detail = detail_resp.json()
    except requests.RequestException as exc:
        print(f"Volume lookup failed for '{query[:40]}': {exc}")
        detail = {}
    detail_info = detail.get("volumeInfo", {})
    record["dimensions"] = detail_info.get("dimensions", {})
    if not record["categories"]:
        record["categories"] = detail_info.get("categories", [])
    with open(cache_path, "w", encoding="utf-8") as fout:
        json.dump(record, fout, indent=2)
    return record


def text_inside_book(book_polygon: np.ndarray, entry: Dict) -> bool:
    path = MplPath(book_polygon)
    return bool(path.contains_point(entry["centroid"]))


def format_rect(rect: np.ndarray) -> str:
    return "; ".join(f"({x:.1f},{y:.1f})" for x, y in rect)


def flatten_authors(authors: Iterable[str]) -> str:
    return ", ".join(authors) if authors else ""


def flatten_isbn(entries: Iterable[Dict]) -> str:
    parts = []
    for item in entries or []:
        identifier = item.get("identifier", "")
        prefix = item.get("type", "")
        parts.append(f"{prefix}:{identifier}" if prefix else identifier)
    return "; ".join(filter(None, parts))


def flatten_dimensions(dimensions: Dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in (dimensions or {}).items())


def flatten_genres(categories: Iterable[str]) -> str:
    return "; ".join(categories) if categories else ""


def parse_height_value(height_value) -> float | None:
    """Normalize Google Books height strings to centimeters."""
    if height_value in (None, ""):
        return None
    if isinstance(height_value, (int, float)):
        return float(height_value)
    if not isinstance(height_value, str):
        return None
    normalized = height_value.replace(",", ".")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", normalized)
    if not match:
        return None
    magnitude = float(match.group(1))
    lower = height_value.lower()
    if "mm" in lower or "millimeter" in lower:
        magnitude /= 10
    elif any(unit in lower for unit in ("cm", "centimeter", "centimeters")):
        pass
    elif any(unit in lower for unit in ("inch", "inches", "in")):
        magnitude *= 2.54
    elif "meter" in lower and "centimeter" not in lower and "millimeter" not in lower:
        magnitude *= 100
    return magnitude


def extract_catalog_height(dimensions: Dict) -> float | None:
    if not isinstance(dimensions, dict):
        return None
    return parse_height_value(dimensions.get("height"))


def process_image(
    image_path: str, di_client: DocumentIntelligenceClient | None, books_api_key: str
) -> List[Dict]:
    book_detections = detect_books(image_path)
    if not book_detections:
        return []
    ocr_entries = ocr(image_path, di_client)
    if not ocr_entries:
        return []

    rows = []
    for idx, detection in enumerate(book_detections):
        relevant_texts = [
            entry for entry in ocr_entries if text_inside_book(detection["polygon"], entry)
        ]
        if not relevant_texts:
            continue
        combined_text = " ".join(entry["text"] for entry in relevant_texts).strip()
        if not combined_text:
            continue
        metadata = search(combined_text, books_api_key)
        dimensions = metadata.get("dimensions", {})
        rows.append(
            {
                "image": Path(image_path).name,
                "book_index": idx,
                "confidence": detection["confidence"],
                "min_rect": format_rect(detection["min_rect"]),
                "optical_height": longest_edge(detection["min_rect"]),
                "identified_text": combined_text,
                "title": metadata.get("title", ""),
                "authors": flatten_authors(metadata.get("authors", [])),
                "genre": flatten_genres(metadata.get("categories", [])),
                "isbn": flatten_isbn(metadata.get("isbn", [])),
                "dimensions": flatten_dimensions(dimensions),
                "catalog_height_cm": extract_catalog_height(dimensions),
            }
        )
    return rows


def plot_height_scatter(df: pd.DataFrame) -> None:
    if "catalog_height_cm" not in df.columns:
        return
    output_dir = Path("book_detections")
    output_dir.mkdir(exist_ok=True)
    for image_name, group in df.groupby("image"):
        subset = group.dropna(subset=["catalog_height_cm", "optical_height"])
        if subset.empty:
            continue
        plt.figure()
        plt.scatter(subset["catalog_height_cm"], subset["optical_height"])
        plt.xlabel("Catalog height (cm)")
        plt.ylabel("Optical height (px)")
        plt.title(f"{image_name} height comparison")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / f"{Path(image_name).stem}_height_scatter.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved height scatter: {plot_path}")


def main(target):
    images = gather_image_paths(target)
    di_client, books_key = ensure_client()
    iterator = tqdm(images, desc="Processing images") if len(images) > 1 else images

    all_rows: List[Dict] = []
    for image_path in iterator:
        all_rows.extend(process_image(image_path, di_client, books_key))

    if not all_rows:
        print("No OCR text matched any detected book masks.")
        return

    df = pd.DataFrame(all_rows)
    # plot_height_scatter(df)
    display_cols = ["image", "optical_height", "genre", "title", "authors", "identified_text"]
    display_df = df[display_cols]
    with pd.option_context("display.max_colwidth", 80):
        print(display_df.to_string(index=False))
    return display_df


def assign_shelves(df: pd.DataFrame, n_shelves: int):
    """
    Cluster books into shelves using the priority rules:
        1. Same author should be close.
        2. Same genre should be close.
        3. Similar optical_height should be close.

    Uses BisectingKMeans on an encoded feature matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ['authors', 'genre', 'optical_height']
    n_shelves : int
        Desired number of shelves

    Returns
    -------
    dict
        {shelf_id: list_of_row_indices}
    """
    if len(df) < n_shelves:
        return {i: [idx] for i, idx in enumerate(df.index)}

    authors = df['authors'].fillna("(unknown_author)")
    genres = df['genre'].fillna("(unknown_genre)")

    # Categorical encoding (authors > genre priority implemented via scaling)
    author_ids = authors.astype('category').cat.codes.values
    genre_ids = genres.astype('category').cat.codes.values

    # Normalize height to mean 0 / std 1
    heights = df['optical_height'].fillna(df['optical_height'].median())
    heights = (heights - heights.mean()) / (heights.std() + 1e-6)

    # ----------- 1. Build feature matrix with priority weighting -----------
    # Strong weight for author, medium for genre, small for height
    X = np.vstack([
        author_ids * 5.0,      # author has highest priority
        genre_ids * 2.0,      # genre second
        heights * 1.0          # height last
    ]).T

    # ----------- 2. Run Bisecting KMeans clustering -----------
    model = BisectingKMeans(n_clusters=n_shelves, random_state=42)
    labels = model.fit_predict(X)

    # ----------- 3. Build shelf dictionary -----------
    shelves = {i: [] for i in range(n_shelves)}
    for idx, cluster_id in enumerate(labels):
        shelves[cluster_id].append(idx)

    return shelves


if __name__ == "__main__":
    import sys
    df = main(sys.argv[1])
    print(assign_shelves(df, 6))
