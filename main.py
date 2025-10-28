import json
import os
import pandas as pd
import requests
import re
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import hashlib
import sys
from glob import glob
from tqdm import tqdm
import warnings
from sklearn.metrics.pairwise import paired_distances

model = YOLO("yolo11n-seg.pt", task="segment")  # COCO-pretrained, includes "book"
op = os.path


def search(query: str, key: str) -> dict:
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
    result = response.json()

    item = result.get("items", [False])[0]
    if not item:
        print(f"No results found for {query}")
        return {}
    rec = {'id': item['id']}
    vinfo = item["volumeInfo"]
    rec["title"] = vinfo["title"]
    rec["authors"] = vinfo.get("authors", [])
    rec['isbn'] = vinfo.get('industryIdentifiers', [])
    resp = requests.get(
        f'https://www.googleapis.com/books/v1/volumes/{item["id"]}',
        params={"projection": "full", "key": key},
    ).json()
    rec['dimensions'] = resp["volumeInfo"].get("dimensions", {})
    return rec


def detect_books(img_path, model):
    results = model(img_path)[0]
    books = []
    book_id = 73  # COCO class ID for "book"
    if results.masks is not None:
        masks = results.masks.data  # (N,H,W)
        classes = results.boxes.cls
        confs = results.boxes.conf
        polygons = results.masks.xy  # polygons

        for m, c, p, box in zip(masks, classes, confs, polygons):
            if c == book_id:
                books.append(box)  # .cpu().numpy())
    return books


def _polygon2bbox(polygons):
    xs = polygons[:, ::2]
    ys = polygons[:, 1::2]
    x0, x1 = xs.min(axis=1), xs.max(axis=1)
    y0, y1 = ys.min(axis=1), ys.max(axis=1)
    return np.c_[x0, y0, x1, y1]


def show_longest_edges(img_path: str, boxes) -> None:
    """
    Display `img_path` with the longest side of each bounding box drawn in green.

    Parameters
    ----------
    img_path : str
        Filesystem path to the image to render.
    boxes : array-like
        Sequence of bounding boxes in (x1, y1, x2, y2) order.
    """
    arr = np.asarray(boxes, dtype=float)
    if arr.size == 0:
        raise ValueError("boxes must contain at least one bounding box.")
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("boxes must be a 2D array with shape (N, 4).")

    img = plt.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")

    for x1, y1, x2, y2 in arr:
        width = x2 - x1
        height = y2 - y1
        if width >= height:
            ax.plot([x1, x2], [y1, y1], color="lime", linewidth=2)
        else:
            ax.plot([x1, x1], [y1, y2], color="lime", linewidth=2)

    fig.tight_layout()
    plt.show()


def match_text_to_books(book_boxes: np.ndarray, text_boxes: np.ndarray) -> np.ndarray:
    """
    For each text_box, find the book_box with which it has the largest intersection area.
    Returns an array of length len(text_boxes), with -1 if no intersection.
    """
    x0b, y0b, x1b, y1b = book_boxes.T
    x0t, y0t, x1t, y1t = text_boxes.T

    # Compute pairwise intersection coordinates
    ix0 = np.maximum(x0t[:, None], x0b)
    iy0 = np.maximum(y0t[:, None], y0b)
    ix1 = np.minimum(x1t[:, None], x1b)
    iy1 = np.minimum(y1t[:, None], y1b)

    # Intersection width/height (clipped at 0)
    iw = np.clip(ix1 - ix0, 0, None)
    ih = np.clip(iy1 - iy0, 0, None)
    inter_area = iw * ih  # (num_texts, num_books)

    # Assign book with max intersection area, or -1 if none
    best = inter_area.argmax(1)
    best[inter_area.max(1) == 0] = -1
    return best


def ocr(img_path, di_client=None, cache=True):
    checksum = md5(img_path)
    cache_path = op.join("ocr_cache", f"{checksum}.json")
    boxes, texts = [], []
    if not op.exists(cache_path):
        if di_client is None:
            raise ValueError("di_client must be provided if no cache exists.")
        with open(img_path, "rb") as f:
            poller = di_client.begin_analyze_document("prebuilt-read", body=f)
        result = poller.result()
        if cache:
            with open(cache_path, "w") as fout:
                json.dump(result.as_dict(), fout, indent=4)
    else:
        with open(cache_path, "r") as fin:
            result = AnalyzeResult(json.load(fin))
    for para in result.paragraphs:
        texts.append(para.content)
        boxes.append(para.bounding_regions[0].polygon)
    return _polygon2bbox(np.array(boxes)), texts


def md5(path):
    with open(path, "rb") as fin:
        content = fin.read()
    checksum = hashlib.md5()
    checksum.update(content)
    return checksum.hexdigest()


def get_longest_edge(polygon):
    paired = np.c_[polygon, np.roll(polygon, -1, axis=0)]
    dist = paired_distances(paired[:, :2], paired[:, 2:], metric="euclidean")
    return paired[dist.argmax()]


def analyze_image(img_path, cache=True):
    checksum = md5(img_path)
    cache_path = op.join("result_cache", f"{checksum}.json")
    if op.exists(cache_path):
        return pd.read_json(cache_path)
    book_polygons = detect_books(img_path, model)
    longest_edges = map(get_longest_edge, book_polygons)
    img = plt.imread(img_path)
    plt.imshow(img)
    for edge in longest_edges:
        plt.plot(edge[::2], edge[1::2], color="lime", linewidth=2)
    for poly in book_polygons:
        edges = np.c_[poly, np.roll(poly, -1, axis=0)]
        for edge in edges:
            plt.plot(edge[::2], edge[1::2], color="red", linestyle="dashed", linewidth=1)
    plt.show()

    # heights = np.maximum(
    #     book_boxes[:, 3] - book_boxes[:, 1], book_boxes[:, 2] - book_boxes[:, 0],
    # )
    # show_longest_edges(img_path, book_boxes)
    # text_boxes, texts = ocr(img_path, di_client)
    # assigned = match_text_to_books(book_boxes, text_boxes)
    # df = pd.DataFrame({"text": texts, "book_id": assigned})
    # grouped_texts = df.groupby("book_id")["text"].apply(lambda x: " ".join(x))
    # grouped_texts = grouped_texts[grouped_texts.index >= 0]
    # heights = heights[grouped_texts.index]
    # df = []
    # for text in grouped_texts:
    #     df.append(search(text, key))
    # df = pd.DataFrame.from_records(df)
    # dims = pd.json_normalize(df.pop('dimensions'))
    # print(dims)
    # if not dims.empty:
    #     df = pd.concat((df, dims), axis=1)
    # else:
    #     df['height'] = np.nan
    # df['pixheight'] = heights
    # if cache:
    #     df.to_json(cache_path, orient="records", indent=2)
    # assert "height" in df
    # return df


def plot_height_regression(df: pd.DataFrame, plot: bool = True):
    """Fit height vs pixheight regression, optionally drawing the scatter and fit line."""
    if df is None:
        raise ValueError("df must be a pandas DataFrame.")

    working = df.loc[:, ["title", "height", "pixheight"]].copy()
    working = working.dropna(subset=["height", "pixheight"])
    if working.empty:
        warnings.warn("No valid rows with both height and pixheight available.")
        return

    def _parse_height(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"[-+]?\d*\.?\d+", value)
            if match:
                return float(match.group())
        return np.nan

    working["pixheight"] = pd.to_numeric(working["pixheight"], errors="coerce")
    working = working.dropna(subset=["pixheight"])
    working["height_numeric"] = working["height"].apply(_parse_height)
    working = working.dropna(subset=["height_numeric"])
    if working.empty:
        raise ValueError("Height values could not be parsed into numeric form.")

    x = working["pixheight"].astype(float).to_numpy()
    y = working["height_numeric"].to_numpy()
    s = working['title']

    fig = None
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(x, y, label="data", color="tab:blue")
        for xx, yy, ss in zip(x, y, s):
            ax.text(xx, yy, ss)

    coeffs = None
    if len(working) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        if plot:
            order = np.argsort(x)
            x_sorted = x[order]
            ax.plot(
                x_sorted,
                slope * x_sorted + intercept,
                color="tab:orange",
                label="fit",
            )
        coeffs = (slope, intercept)
    if plot:
        ax.set_xlabel("pixheight")
        ax.set_ylabel("height (numeric)")
        ax.legend()
        ax.set_title("Height vs Pixel Height")
        fig.tight_layout()
    plt.show()
    return fig, coeffs


if __name__ == "__main__":
    if op.exists(".secrets.json"):
        with open(".secrets.json", "r") as fin:
            secrets = json.load(fin)

        credential = AzureKeyCredential(secrets["AZURE_API_KEY"])
        di_client = DocumentIntelligenceClient(secrets["AZURE_DI_ENDPOINT"], credential)
        key = secrets["GOOGLE_BOOKS_API_KEY"]
    else:
        di_client = None
        key = ""

    if op.isfile(sys.argv[1]):
        df = analyze_image(sys.argv[1])
    else:
        for image in tqdm(glob(op.join(sys.argv[1], "*.jpg"))):
            df = analyze_image(image)
