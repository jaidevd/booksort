# Repository Guidelines

## Project Structure & Module Organization
Code lives in two top-level scripts: `main.py` (interactive detection + OCR + metadata lookup) and `detect_and_evaluate.py` (batch evaluation + visualization). Keep reference imagery in `test_images/`, polygon labels in `labels/`, and cached OCR responses in `ocr_cache/`; the scripts autodiscover files by those names. Checkpoint weights such as `yolo11x-seg.pt` stay in the root so runs can load them without extra path plumbing.

## Build, Test, and Development Commands
- `poetry install` — provision the Python 3.12 environment with the CUDA 12.6 Torch wheels provided in `pyproject.toml`.
- `poetry run python main.py <image-or-dir>` — produce detections and cached OCR for a single file or a directory like `test_images`.
- `poetry run python detect_and_evaluate.py --images-dir test_images --labels-dir labels --weights yolo11x-seg.pt` — compute timing + mAP tables and save masks into `book_detections/`.
- `poetry run flake8` — run the lint gate (99 character lines, `T201` relaxed) before opening a PR.

## Coding Style & Naming Conventions
Stick to four-space indentation and snake_case functions (`detect_books`, `plot_height_regression`); reserve CapWords for future classes. Use type hints when adding public functions, prefer f-strings, and keep helper functions near their callers so our single-file scripts remain understandable. Avoid gratuitous module churn—new modules should exist only when logic is reused by both primary scripts.

## Testing & Evaluation Guidelines
There is no pytest harness, so treat `detect_and_evaluate.py` as the regression suite. Run it whenever you change model loading, mask post-processing, or polygon parsing, and inspect the saved overlays in `book_detections/<weight>/<threshold>/` before merging. When altering OCR or metadata aggregation, clear only the affected JSON files inside `ocr_cache/` and document the manual checks you ran in the PR description.

## Commit & Pull Request Guidelines
The git history uses short uppercase prefixes like `ENH: Cached OCR and analysis results` and `WIP: Better book detection`; continue with tags such as `ENH`, `FIX`, `DOC`, or `WIP` plus an imperative summary. Pull requests should include a concise scope paragraph, exact reproduction commands, expected artifacts (mAP numbers, overlay paths, or screenshots), and any required datasets or secrets.

## Security & Configuration Tips
Store `AZURE_API_KEY`, `AZURE_DI_ENDPOINT`, and `GOOGLE_BOOKS_API_KEY` in an untracked `.secrets.json`—`main.py` reads it directly, so leaking it would compromise both services. Keep YOLO weights out of version control; symlink or document their source instead. If a change adds new credentials or large assets, call that out in the PR so reviewers know how to reproduce safely.
