# RotomCV: Computer Vision Pokemon Card Scanner
**Goal:** Build a production-grade Computer Vision Pipeline to digitize physical Pokemon cards from raw photos into structured data.

**Current State:** Phase 5 (Database Integration) Complete.

## Features
-   **Computer Vision Pipeline**: Automated rectification and cropping of card zones to a standardized 600x840 "Digital Twin".
-   **Hybrid OCR Engine**: Combines PaddleOCR (v4 Mobile) and Tesseract for maximum accuracy on card names and numbers.
-   **MongoDB Integration**: Scalable backend storing 20,000+ cards and scan logs, enabling real-time lookups.
-   **Smart Matching**: Fuzzy logic identifies cards even with partial OCR errors (e.g. "Serperiorex" -> "Serperior EX").
-   **Ground Truth Annotator**: Custom GUI for creating training data and validating crops.

## 🏗 Core Architecture
The project is built as a modular pipeline where data flows through specialized services in `src/rotomcv/`:

| Component | File | Responsibility | Key Tech | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Annotator** | `annotator.py` | GUI for creating Ground Truth data (corner labels). | Tkinter | Stable |
| **Rectifier** | `rectifier.py` | Unwarps raw photos to a standard 600x840 geometry. | OpenCV | Critical (Foundation) |
| **ROI Explorer** | `roi_explorer.py` | Slices the standard card into zones (Name, Number, Set). | NumPy (In-Memory) | Optimized (No Disk I/O) |
| **OCR Service** | `ocr_service.py` | Reads text from slices. | PaddleOCR (Mobile v4) | Optimized (~1.6s/ROI) |
| **Database** | `db_client.py` | Manages card data and scan logs. | MongoDB + PyMongo | **New!** (Scalable) |
| **Orchestrator** | `pipeline_manager.py` | Runs the batch process across the dataset. | tqdm, CSV, MongoDB | Production Ready |

## Setup
1.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Install Tesseract OCR:**
    -   Windows: `scoop install tesseract` (or download installer)
    -   Ensure `tesseract` is in your system PATH.
    -   Download `eng.traineddata` (Best version) to `tessdata`.
3.  **MongoDB Setup:**
    -   Install MongoDB Community Edition.
    -   Ensure it is running on `localhost:27017` (or set `MONGO_URI` in `.env`).
    -   Run migration execution: `python src/rotomcv/ingest_cards.py`

## Usage
### 1. Run Hybrid OCR Pipeline
Processing images -> OCR Text -> MongoDB Log:
```bash
python src/rotomcv/pipeline_manager.py
```

### 2. Enrich & Match Results (Legacy)
Merging OCR text with Local Database -> Final JSON:
```bash
python enrich_results.py
```
Output: `data/enriched_results.json`

### Annotator Keybinds:
| Action | Key | Description |
| :--- | :--- | :--- |
| Confirm / Next | Enter | Validate & save |
| Undo | Backspace | Remove last point |
| Skip | Esc | Skip current image |
| Quit | Q | End program |
| Reset | R | Clear all corners |

## 💾 Data Management
-   **Golden Master:** `data/manifests/dataset.csv`
    -   The source of truth for file paths and verified corner coordinates.
-   **Pipeline Output:**
    -   **CSV:** `data/pipeline_output/pipeline_results.csv` (Atomic backup)
    -   **MongoDB:** `rotomcv.scans` collection (Primary log)

## 🗺 Current Status & Roadmap

### ✅ Phase 1-5: Core Pipeline & Database
The backend logic is fully functional, optimized, and verified.
-   **Annotator**: Stable GUI for training data.
-   **Rectifier**: Solid OpenCV foundations.
-   **OCR**: Tuned PulseOCR + Tesseract hybrid.
-   **MongoDB**: Migrated 20,000+ cards from flat JSON files to a local MongoDB instance.
-   **Scan Logging**: Pipeline automatically logs results to the database.

### 🚧 Phase 6: Automatic Detection
Replace manual corner annotation with an Automatic Card Detector.
-   **Current Status**: Implemented but performance is suboptimal. Needs retraining or a switch to YOLOv8.
-   **Goal**: Find card edges instantly in a camera frame.

### 🔮 Phase 7: The "Live" API & Frontend (Future)
-   **FastAPI Backend**: POST /scan -> Image to JSON.
-   **Frontend**: React/Next.js UI for point-and-scan functionality.
-   **Collection Management**: Save scans to user "Binder" folders.

### Visual Debug Example
![Serperior Debug Extraction from rectified image](data/serperior_debug/debug_ph_0029.jpg)
