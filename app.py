from typing import Union, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import traceback

from llama_wrapper import LlamaWrapper
from language_translator import LangTranslate
from custom_decorators import log_runtime_metrics
import os
from io import BytesIO
import base64
from PIL import Image

origins = [
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:3000"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#model_name = 'TheBloke/CodeLlama-7B-Instruct-GGUF'
model_name = 'TheBloke/Mistral-7B-v0.1-GGUF/mistral-7b-v0.1.Q5_K_M.gguf'
# model_name = 'TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_K_M.gguf'
# model_name = 'TheBloke/Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q4_K_M.gguf'

llama_wrapper = LlamaWrapper(model_name)

ENABLE_LANGTRANSLATE = os.environ.get('ENABLE_LANGTRANSLATE', False)

lang_translate = LangTranslate() if ENABLE_LANGTRANSLATE else None

class GenerateItem(BaseModel):
    input_texts: list
    stop_sequences: Optional[str] = None
    temperature: Optional[float] = 0.7
    logprobs: Optional[float] = None

class TranslateItem(BaseModel):
    input_texts: list
    src_lang: Optional[str] = 'en'
    tgt_lang: Optional[str] = 'en'

@app.post('/generate')
@log_runtime_metrics
def generate_text(data: GenerateItem):
    input_texts = data.input_texts
    stop_sequences = data.stop_sequences
    temperature = data.temperature
    logprobs = data.logprobs

    try:
        response = llama_wrapper.generate_text(input_texts, temperature, stop_sequences)
        print (response)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, description="An error occurred during text generation.")
    
    return response

@app.post('/translate')
@log_runtime_metrics
def translate_text(item: TranslateItem):
    input_texts = item.input_texts
    src_lang = item.src_lang
    tgt_lang = item.tgt_lang

    try:
        response = lang_translate.translate_paragraph(input_texts, src_lang, tgt_lang)
    except Exception as e:
        print (f'Error {str(e)}')
        print (traceback.format_exc())
        return {"description":"An error occurred during text translation."}, 400

    return response

class ResizeImageItem(BaseModel):
    base64_image: str
    ratio: Optional[float] = 1
    quality: Optional[int] = 80

@app.post('/resize_image')
def resize_image(item: ResizeImageItem):
    # Decode base64 image
    image_data = base64.b64decode(item.base64_image)
    # Open image using Pillow
    img = Image.open(BytesIO(image_data))
    
    # Convert image to RGB if it is RGBA
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize image
    width, height = img.size
    ratio = item.ratio
    resized_img = img.resize((int(width*ratio), int(height*ratio)))
    # Convert image to bytes
    buffer = BytesIO()
    resized_img.save(buffer, format='JPEG', quality=item.quality)
    # Encode image as base64
    resized_base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {'resized_image': resized_base64_image}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5000)