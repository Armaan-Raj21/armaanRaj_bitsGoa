# Bill Extraction API

## Overview
This API extracts line items, totals, and invoice details from PDF and Image bills using Generative AI.

## Solution Logic
1.  **Input:** Accepts a public URL of a bill (PDF/Image).
2.  **Processing:** 
    - Downloads the file.
    - If PDF, converts pages to images using `pdf2image`.
    - Prepares images for the Vision Model.
3.  **AI Analysis:** Uses **Gemini 1.5 Flash** to analyze visual data, identifying line items while strictly avoiding double-counting (e.g., ignoring sub-total summaries if line items are present).
4.  **Output:** Returns JSON strictly formatted to the HackRx schema.

## Tech Stack
- Python (FastAPI)
- Google Gemini 1.5 Flash (Vision)
- Render (Deployment)

## API Endpoint
`POST /extract-bill-data`
