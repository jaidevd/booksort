from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from main import (
    SUPPORTED_EXTS,
    assign_shelves,
    ensure_client,
    process_image,
)

app = FastAPI(
    title="Booksort Shelving API",
    description=(
        "Upload bookshelf images to detect, identify, and cluster books into shelves. "
        "The number of uploaded images defines the number of shelves."
    ),
    version="0.1.0",
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
DI_CLIENT, BOOKS_API_KEY = ensure_client()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _persist_upload(upload: UploadFile, contents: bytes) -> str:
    """Persist an uploaded image to a temporary file on disk."""
    suffix = Path(upload.filename or "").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        return tmp.name


def _validate_uploads(files: List[UploadFile]) -> None:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix and suffix not in SUPPORTED_EXTS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTS)}",
            )


@app.post("/shelves")
async def assign_shelves_from_images(files: List[UploadFile] = File(...)) -> Dict:
    """
    Cluster all detected books into shelves based on uploaded bookshelf images.

    The number of uploaded images determines the number of shelves returned.
    """

    _validate_uploads(files)
    n_shelves = len(files)
    all_rows: List[Dict] = []
    temp_paths: List[str] = []

    try:
        for upload in files:
            contents = await upload.read()
            if not contents:
                raise HTTPException(
                    status_code=400, detail=f"Empty upload for '{upload.filename}'."
                )
            temp_path = _persist_upload(upload, contents)
            temp_paths.append(temp_path)

            rows = process_image(temp_path, DI_CLIENT, BOOKS_API_KEY)
            for row in rows:
                row["image"] = upload.filename or row.get("image", Path(temp_path).name)
            all_rows.extend(rows)
    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    if not all_rows:
        raise HTTPException(
            status_code=404,
            detail="No books detected or OCR text matched any masks in the provided images.",
        )

    df = pd.DataFrame(all_rows)
    shelf_map = assign_shelves(df, n_shelves)
    records = df.to_dict(orient="records")

    shelves: Dict[str, List[Dict]] = {}
    for shelf_id, indices in shelf_map.items():
        shelves[str(shelf_id)] = [records[idx] for idx in indices]

    return {
        "shelf_count": n_shelves,
        "total_books": len(records),
        "books": records,
        "shelves": shelves,
    }


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the mobile-friendly uploader page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)
