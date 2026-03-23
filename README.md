# Invoice DocAI

Comparative analysis of OCR-based and end-to-end transformer approaches
for key information extraction from scanned receipts.

## Overview

This project evaluates two paradigms for extracting structured fields
(vendor name, date, total amount) from receipt images on the
ICDAR 2019 SROIE benchmark:

1. **OCR Baseline** -- EasyOCR (CRAFT + CRNN) with rule-based field extraction
2. **Donut Fine-tuned** -- end-to-end vision-language transformer (Swin + BART),
   fine-tuned on SROIE training data

Both pipelines are evaluated under clean and messenger-grade corrupted
image conditions. A 12-category error taxonomy and cross-pipeline
correlation analysis are provided.

## Results (quick mode, 80 validation receipts)

| Pipeline   | Condition | Vendor F1 | Date F1 | Total F1 | Micro F1 |
|------------|-----------|-----------|---------|----------|----------|
| OCR        | clean     | 0.49      | 0.78    | 0.63     | 0.63     |
| OCR        | corrupted | 0.40      | 0.56    | 0.45     | 0.47     |
| Donut-FT   | clean     | 0.82      | 0.63    | 0.78     | 0.75     |
| Donut-FT   | corrupted | 0.69      | 0.54    | 0.64     | 0.63     |

Key findings:

- Donut-FT achieves higher overall F1 (0.75 vs 0.63)
- OCR wins on date extraction (F1 = 0.78 vs 0.63) due to regex precision
- Donut-FT is more robust to degradation (F1 drop: -0.12 vs -0.17)
- Low cross-pipeline error correlation (r = 0.30) suggests ensemble potential

## Project structure

```
invoice_docai/
  v2/
    RUN_ALL_COLAB.ipynb          -- master notebook (full pipeline)
    src/docai_utils.py           -- utility library (595 lines)
    notebooks/
      01_prepare_sroie.ipynb     -- dataset preparation
      02_baseline_ocr.ipynb      -- OCR baseline evaluation
      03_donut_inference.ipynb   -- pretrained Donut inference
      03b_donut_finetune.ipynb   -- Donut fine-tuning
      04_robustness_eval.ipynb   -- robustness evaluation
      05_summary.ipynb           -- summary and visualization
      06_deep_error_analysis.ipynb
      07_literature_and_why.ipynb
    outputs/                     -- CSV predictions and PNG figures
  data/sroie/
    processed/                   -- train/val manifests (CSV, JSONL)
  PROJECT_REPORT.md              -- detailed technical report (Russian)
```

## Requirements

- Python 3.10+
- Google Colab with T4 GPU (16 GB VRAM) or equivalent
- See `v2/requirements.txt` for package dependencies

## Dataset

SROIE 2019 (ICDAR). The raw images are not included in this repository.
Download from the competition page or use the Google Drive link in the
master notebook. Processed manifests are provided in `data/sroie/processed/`.

## Usage

Open `v2/RUN_ALL_COLAB.ipynb` in Google Colab and run all cells.
The notebook handles dataset download, OCR inference, Donut fine-tuning,
robustness evaluation, and visualization.

Individual pipeline stages can be run separately via notebooks in `v2/notebooks/`.

## Technology stack

- Python 3.12, PyTorch 2.x (FP16 mixed precision)
- EasyOCR 1.7 (CRAFT + CRNN)
- Hugging Face Transformers (Donut VisionEncoderDecoderModel)
- OpenCV (image processing, corruption simulation)
- pandas, NumPy, Matplotlib

## Limitations

- Quick mode only: 240 train / 80 val documents, 2 training epochs
- No layout features (bounding box coordinates not used)
- Rule-based extraction (no trained NER for the OCR pipeline)
- Results are lower bounds; full training would close the gap with SOTA

## Author

Sergei Solovev, Faculty of Informatics, Mathematics, and Computer Science,
HSE University, Moscow.

## License

MIT
