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

# Get API Key
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")

class DocumentRequest(BaseModel):
    document: str

def download_file(url):
    try:
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

def image_to_base64(image):
    buffered = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_gemini_smart(image_b64):
    # LIST OF MODELS TO TRY (In order of preference)
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro",
        "gemini-pro-vision"
    ]
    
    headers = {'Content-Type': 'application/json'}
    
    prompt_text = """
    Extract bill data. Return JSON.
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

    last_error = ""

    # LOOP THROUGH MODELS
    for model_name in models_to_try:
        print(f"Attempting to use model: {model_name}...")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GENAI_API_KEY}"
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            # If success (200), return immediately
            if response.status_code == 200:
                print(f"SUCCESS with {model_name}")
                return response.json()
            else:
                error_msg = response.text
                print(f"Failed {model_name}: {error_msg}")
                last_error = error_msg
        except Exception as e:
            print(f"Connection error on {model_name}: {e}")

    # If all failed
    raise Exception(f"All models failed. Last error: {last_error}")

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

        # 3. Base64
        img_b64 = image_to_base64(target_image)

        # 4. Call Smart AI
        raw_response = call_gemini_smart(img_b64)
        
        # 5. Parse
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