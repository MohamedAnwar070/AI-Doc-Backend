# OCR Engine Output Comparison

This document provides a concise comparison of the OCR engines evaluated based strictly on their output quality and characteristics.

### Engines Evaluated
`google_docai` | `sur_ocr` | `ocr_open_ai` | `olm_ocr`

---

### 1. google_docai
* **Text Detection:** Excellent and highly accurate.
* **Layouts:** Poor layout retention; structural alignment is frequently bad or completely missing in the final output.

### 2. sur_ocr
* **Strengths:** Good at generating bounding boxes (bbox), highlighted OCR overlays, and extracting individual line coordinates.
* **Weaknesses:** Not suitable for generating full, structured Markdown documents.

### 3. ocr_open_ai
* **Performance:** The best overall among all tested engines.
* **Note:** This is a paid service and requires specific API configuration details.

### 4. olm_ocr
* **Performance:** Needs a GPU to be evaluated perfectly and reliably. 
* **Note:** High hardware requirements (GPU/VRAM) to function at its full potential for layout extraction.
