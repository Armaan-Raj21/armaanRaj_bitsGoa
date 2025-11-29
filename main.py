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
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

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

def get_generative_model_name():
    """
    Asks Google for models that specifically support 'generateContent'.
    Ignores embedding models.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GENAI_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            # Fallback if list fails
            return "models/gemini-1.5-flash"

        data = response.json()
        
        # 1. Filter: Only keep models that can GENERATE content
        valid_models = []
        for m in data.get('models', []):
            methods = m.get('supportedGenerationMethods', [])
            if 'generateContent' in methods:
                valid_models.append(m['name'])
        
        print(f"VALID MODELS: {valid_models}")
        
        if not valid_models:
            raise Exception("No generative models available for this API Key.")

        # 2. Priority: Try to find the best Gemini model
        priorities = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision", "gemini-1.0-pro"]
        
        for p in priorities:
            for vm in valid_models:
                if p in vm:
                    return vm
        
        # 3. Fallback: Just take the first one that works
        return valid_models[0]
        
    except Exception as e:
        print(f"Model selection error: {e}")
        return "models/gemini-1.5-flash"

def call_gemini_final(image_b64):
    # 1. Get a valid model name
    model_name = get_generative_model_name()
    short_name = model_name.replace("models/", "")
    
    print(f"USING MODEL: {short_name}")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{short_name}:generateContent?key={GENAI_API_KEY}"
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
        raise Exception(f"Google API Error ({short_name}): {response.text}")
        
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

        # 3. Base64
        img_b64 = image_to_base64(target_image)

        # 4. Call AI
        raw_response = call_gemini_final(img_b64)
        
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
        print(f"CRITICAL ERROR: {str(e)}")
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