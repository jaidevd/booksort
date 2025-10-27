import numpy as np
import json
import os.path as op
import pandas as pd
import requests
from ultralytics import YOLO
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import hashlib

model = YOLO("yolov8n-seg.pt")  # COCO-pretrained, includes "book"


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
    return response.json()


def detect_books(img_path, model):
    results = model(img_path)[0]
    books = []
    if results.masks is not None:
        masks = results.masks.data  # (N,H,W)
        classes = results.boxes.cls
        confs = results.boxes.conf
        boxes = results.boxes.xyxy  # Bounding boxes

        book_id = 73  # COCO class ID for "book"
        for m, c, p, box in zip(masks, classes, confs, boxes):
            if c == book_id and p > 0.5:
                books.append(box.cpu().numpy())
    return np.array(books)


def _polygon2bbox(polygons):
    xs = polygons[:, ::2]
    ys = polygons[:, 1::2]
    x0, x1 = xs.min(axis=1), xs.max(axis=1)
    y0, y1 = ys.min(axis=1), ys.max(axis=1)
    return np.c_[x0, y0, x1, y1]


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
    cache_path = img_path + ".json"
    boxes, texts = [], []
    if not op.exists(cache_path):
        if di_client is None:
            raise ValueError("di_client must be provided if no cache exists.")
        with open(img_path, "rb") as f:
            poller = di_client.begin_analyze_document("prebuilt-read", body=f)
        result = poller.result()
        if cache:
            with open(img_path + ".json", "w") as fout:
                json.dump(result.as_dict(), fout, indent=4)
    else:
        with open(cache_path, "r") as fin:
            result = AnalyzeResult(json.load(fin))
    for para in result.paragraphs:
        texts.append(para.content)
        boxes.append(para.bounding_regions[0].polygon)
    return _polygon2bbox(np.array(boxes)), texts

def sha256(content):
    checksum = hashlib.md5()
    checksum.update(content)
    return checksum.hexdigest()


if __name__ == "__main__":
    key = "API_KEY_HERE"
    if op.exists(".secrets.json"):
        with open(".secrets.json", "r") as fin:
            secrets = json.load(fin)

        credential = AzureKeyCredential(secrets["AZURE_API_KEY"])
        di_client = DocumentIntelligenceClient(secrets["AZURE_DI_ENDPOINT"], credential)
    else:
        di_client = None

    book_boxes = detect_books("test.jpg", model)
    text_boxes, texts = ocr("test.jpg", di_client)
    assigned = match_text_to_books(book_boxes, text_boxes)
    df = pd.DataFrame({"text": texts, "book_id": assigned})
    grouped_texts = df.groupby("book_id")["text"].apply(lambda x: " ".join(x))
    grouped_texts = grouped_texts[grouped_texts.index >= 0]
    for text in grouped_texts:
        result = search(text, key)
        item = result.get("items", [False])[0]
        if not item:
            print(f"No results found for {text}")
            continue
        vinfo = item["volumeInfo"]
        print(vinfo["title"], vinfo.get("authors", []))
        resp = requests.get(
            f'https://www.googleapis.com/books/v1/volumes/{item["id"]}',
            params={"projection": "full", "key": key},
        ).json()
        print(resp['volumeInfo'].get('dimensions', {}))
