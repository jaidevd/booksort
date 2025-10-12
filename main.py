from ultralytics import YOLO
import numpy as np
import json
import os.path as op
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

model = YOLO("yolov8n-seg.pt")  # COCO-pretrained, includes "book"


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
                books.append(box.numpy())
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


def ocr(img_path, di_client, cache=True):
    cache_path = img_path + ".json"
    boxes, texts = [], []
    if not op.exists(cache_path):
        with open(img_path, "rb") as f:
            poller = di_client.begin_analyze_document("prebuilt-read", body=f)
        result = poller.result()
        if cache:
            with open(img_path + ".json", "w") as fout:
                json.dump(result.as_dict(), fout, indent=4)
    else:
        with open(cache_path, "r") as fin:
            result = AnalyzeResult(json.load(fin))
    print(len(result.paragraphs))
    for para in result.paragraphs:
        texts.append(para.content)
        boxes.append(para.bounding_regions[0].polygon)
    print(len(boxes), len(texts))
    return _polygon2bbox(np.array(boxes)), texts


if __name__ == "__main__":

    with open(".secrets.json", "r") as fin:
        secrets = json.load(fin)

    credential = AzureKeyCredential(secrets["AZURE_API_KEY"])
    di_client = DocumentIntelligenceClient(secrets["AZURE_DI_ENDPOINT"], credential)

    book_boxes = detect_books("test.jpg", model)
    text_boxes, texts = ocr("test.jpg", di_client)
    assigned = match_text_to_books(book_boxes, text_boxes)
