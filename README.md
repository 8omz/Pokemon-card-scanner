# Pokemon Card Scanner

**Goal**: Build a production-grade system to digitize physical Pokemon cards from raw photos into structured data.

**Current State**: Phase 4 (Batch Processing) Complete. The backend logic is fully functional, optimized, and verified.

---

## 🏗 Core Architecture

The project is built as a modular pipeline where data flows through specialized services in `src/card_annotation_tool/`:

| Component | File | Responsibility | Key Tech | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Annotator** | `annotator.py` | GUI for creating Ground Truth data (corner labels). | Tkinter | Stable |
| **Rectifier** | `rectifier.py` | Unwarps raw photos to a standard **600x840** geometry. | OpenCV | Critical (Foundation) |
| **ROI Explorer** | `roi_explorer.py` | Slices the standard card into zones (Name, Number, Set). | NumPy (In-Memory) | Optimized (No Disk I/O) |
| **OCR Service** | `ocr_service.py` | Reads text from slices. | PaddleOCR (Mobile v4) | Optimized (~1.6s/ROI) |
| **Orchestrator**| `pipeline_manager.py`| Runs the batch process across the dataset. | `tqdm`, CSV | Production Ready |

## 💾 Data Management

*   **Golden Master**: `data/manifests/dataset.csv`
    *   The source of truth for file paths and verified corner coordinates.
*   **Pipeline Output**: `data/pipeline_output/pipeline_results.csv`
    *   The results of the batch run, containing extracted data and confidence flags.

## 🚀 Technical Wins

1.  **Original-Space Truth**: We map everything back to the raw image logic but operate in a standardized "Digital Twin" space (600x840).
2.  **Latency Optimization**:
    *   **Mobile Models**: Switched to `PP-OCRv4` Mobile for a ~30% speedup.
    *   **Zero-Copy**: `roi_explorer` passes NumPy arrays directly to OCR without saving to disk.
    *   **Logic Pruning**: Disabled `angle_classification` because the Rectifier guarantees upright images.
3.  **Robustness**: The pipeline handles missing files, bad paths, and OCR failures without crashing, using atomic CSV writing to save progress.

## 🗺 Future Roadmap (Phase 5+)

The backend logic is ready. The next steps transition the project from a developer tool to a user product.

### Phase 5: The Visual Fail-Safe (Art Matching)
Implement Perceptual Hashing (pHash). If OCR confidence is low, compare the art window hash against a database of known card art to find matches visually.

### Phase 6: The Real-Time Auto-Detector
Replace manual corner annotation with an Automatic Card Detector (OpenCV contours or YOLO) to find card edges instantly in a camera frame.

### Phase 7: The "Live" API & Frontend
*   **FastAPI Backend**: `POST /scan` -> Image to JSON.
*   **Frontend**: React/Next.js UI for point-and-scan functionality.

### Phase 8: Collection Management
*   **Database Integration**: Save scans to "Binder" folders.
*   **Analytics**: Track collection value over time.

---

*This project is a sophisticated computer vision pipeline demonstrating advanced OCR optimization, geometric transformation, and robust software architecture.*
