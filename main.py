import os
import uvicorn
import requests
import google.generativeai as genai
import json
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pdf2image import convert_from_path
from PIL import Image

app = FastAPI()

# Configure Gemini
# We will set this API Key in Render Environment Variables later
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

class DocumentRequest(BaseModel):
    document: str

def download_file(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        
        ext = ".jpg" # Default
        if "pdf" in content_type: ext = ".pdf"
        elif "png" in content_type: ext = ".png"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            return tmp.name, ext
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

@app.post("/extract-bill-data")
async def extract_bill_data(request: DocumentRequest):
    temp_path = None
    try:
        # 1. Download
        temp_path, ext = download_file(request.document)
        
        # 2. Prepare Images for AI
        image_parts = []
        if ext == ".pdf":
            # Convert PDF to images
            try:
                images = convert_from_path(temp_path)
                image_parts.extend(images)
            except Exception as e:
                raise HTTPException(status_code=500, detail="PDF processing failed. Poppler issue.")
        else:
            # Handle standard images
            img = Image.open(temp_path)
            image_parts.append(img)

        # 3. Prompt Engineering
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """
        Analyze these bill images. Extract line item details.
        
        CRITICAL RULES:
        1. Identify the 'page_type': "Bill Detail", "Final Bill", or "Pharmacy".
        2. Extract Item Name, Amount, Rate, and Quantity.
        3. DO NOT DOUBLE COUNT. If a page summarizes previous items, process it but do not add the items again if they were on previous pages.
        4. Calculate 'total_item_count'.
        
        Output strictly valid JSON (no markdown) in this exact structure:
        {
            "pagewise_line_items": [
                {
                    "page_no": "1",
                    "page_type": "Bill Detail",
                    "bill_items": [
                        {
                            "item_name": "Consultation",
                            "item_amount": 500.0,
                            "item_rate": 500.0,
                            "item_quantity": 1.0
                        }
                    ]
                }
            ],
            "total_item_count": 1
        }
        """

        # 4. Generate
        response = model.generate_content([prompt, *image_parts])
        
        # 5. Parse JSON
        try:
            # Strip markdown code blocks if Gemini adds them
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            extracted_data = json.loads(clean_text)
        except:
            # Fallback if AI fails to generate JSON
            extracted_data = {"pagewise_line_items": [], "total_item_count": 0}

        # 6. Return Formatted Response
        return {
            "is_success": True,
            "token_usage": {
                "total_tokens": response.usage_metadata.total_token_count,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            },
            "data": extracted_data
        }

    except Exception as e:
        return {
            "is_success": False,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {"pagewise_line_items": [], "total_item_count": 0},
            "error": str(e)
        }
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
