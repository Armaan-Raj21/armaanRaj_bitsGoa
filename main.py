import os
import uvicorn
import requests
import json
import tempfile
import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pdf2image import convert_from_path
from PIL import Image

app = FastAPI()

GENAI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

class DocumentRequest(BaseModel):
    document: str

def download_file(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        # --- FIX: ROBUST EXTENSION DETECTION ---
        # 1. Check Content-Type header
        content_type = response.headers.get('content-type', '').lower()
        # 2. Check URL string
        url_lower = url.lower()
        
        ext = ".jpg" # Default fallback
        
        # If header says PDF OR url ends with .pdf
        if "pdf" in content_type or url_lower.endswith(".pdf"):
            ext = ".pdf"
        elif "png" in content_type or url_lower.endswith(".png"):
            ext = ".png"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            return tmp.name, ext
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

def image_to_base64(image):
    buffered = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_best_available_model():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GENAI_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return "gemini-1.5-flash"

        data = response.json()
        all_models = data.get('models', [])
        
        valid_candidates = []
        
        for m in all_models:
            name = m['name'].replace("models/", "")
            methods = m.get('supportedGenerationMethods', [])
            if 'generateContent' not in methods: continue
            if 'embedding' in name: continue
            
            is_risky = 'exp' in name or 'preview' in name or '002' in name
            valid_candidates.append((name, is_risky))
            
        if not valid_candidates:
            return "gemini-1.5-flash"

        # 1. Try Safe Flash
        for name, risky in valid_candidates:
            if 'flash' in name and not risky: return name
        # 2. Try Safe Pro
        for name, risky in valid_candidates:
            if 'pro' in name and not risky: return name

        return valid_candidates[0][0]

    except:
        return "gemini-1.5-flash"

def call_gemini_auto(image_b64):
    model_name = get_best_available_model()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GENAI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    prompt_text = """
    Extract bill data strictly as JSON.
    Schema:
    {
        "pagewise_line_items": [
            {
                "page_no": "1",
                "page_type": "Bill Detail",
                "bill_items": [
                    {
                        "item_name": "str",
                        "item_amount": 0.0,
                        "item_rate": 0.0,
                        "item_quantity": 0.0
                    }
                ]
            }
        ],
        "total_item_count": 0
    }
    """
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
            ]
        }]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Google API Error: {response.text}")
    return response.json()

@app.post("/extract-bill-data")
async def extract_bill_data(request: DocumentRequest):
    temp_path = None
    try:
        # 1. Download
        temp_path, ext = download_file(request.document)
        
        # 2. Get Image
        target_image = None
        if ext == ".pdf":
            try:
                images = convert_from_path(temp_path)
                if images: target_image = images[0]
            except:
                raise HTTPException(status_code=500, detail="PDF Error. Is Poppler installed?")
        else:
            target_image = Image.open(temp_path)

        if not target_image: raise Exception("No image found.")

        # 3. Process
        img_b64 = image_to_base64(target_image)
        raw_response = call_gemini_auto(img_b64)
        
        # 4. Parse
        try:
            candidates = raw_response.get('candidates', [])
            if not candidates: raise Exception("AI returned empty candidates")
            
            text_part = candidates[0]['content']['parts'][0]['text']
            clean_text = text_part.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            
            usage = raw_response.get('usageMetadata', {})
            token_usage = {
                "total_tokens": usage.get('totalTokenCount', 0),
                "input_tokens": usage.get('promptTokenCount', 0),
                "output_tokens": usage.get('candidatesTokenCount', 0)
            }
        except:
            data = {"pagewise_line_items": [], "total_item_count": 0}
            token_usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        return {
            "is_success": True,
            "token_usage": token_usage,
            "data": data
        }

    except Exception as e:
        return {
            "is_success": False,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {"pagewise_line_items": [], "total_item_count": 0},
            "error": str(e)
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)