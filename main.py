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

# 1. Setup Gemini
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
    
    # --- DEBUGGING: PRINT AVAILABLE MODELS TO LOGS ---
    try:
        print("---- CHECKING AVAILABLE MODELS ----")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Found model: {m.name}")
        print("---- END MODEL CHECK ----")
    except Exception as e:
        print(f"Error checking models: {e}")

class DocumentRequest(BaseModel):
    document: str

def download_file(url):
    try:
        # Fake user agent to prevent 403 errors from some sites
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        ext = ".jpg"
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
        
        # 2. Prepare Images
        image_parts = []
        if ext == ".pdf":
            try:
                images = convert_from_path(temp_path)
                image_parts.extend(images)
            except Exception as e:
                # If poppler fails, return a clear error
                raise HTTPException(status_code=500, detail="PDF Error. Ensure Poppler is installed.")
        else:
            img = Image.open(temp_path)
            image_parts.append(img)

        # 3. Call AI - USING THE CORRECT MODEL
        # We use 'gemini-1.5-flash' which is the standard current vision model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        Analyze these bill images and extract data.
        Output ONLY valid JSON.
        
        Schema:
        {
            "pagewise_line_items": [
                {
                    "page_no": "1",
                    "page_type": "Bill Detail",
                    "bill_items": [
                        {
                            "item_name": "Item Name",
                            "item_amount": 100.0,
                            "item_rate": 100.0,
                            "item_quantity": 1.0
                        }
                    ]
                }
            ],
            "total_item_count": 1
        }
        """

        response = model.generate_content([prompt, *image_parts])
        
        # 4. Parse Response
        try:
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
        except:
            data = {"pagewise_line_items": [], "total_item_count": 0}

        return {
            "is_success": True,
            "token_usage": {
                "total_tokens": response.usage_metadata.total_token_count,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            },
            "data": data
        }

    except Exception as e:
        # Print the actual error to logs so we can see it
        print(f"ERROR: {str(e)}")
        return {
            "is_success": False,
            "data": {"pagewise_line_items": [], "total_item_count": 0},
            "error": str(e)
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)