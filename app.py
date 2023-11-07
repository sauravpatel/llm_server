from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from flask import Flask, request, jsonify
from flask_cors import CORS

import traceback

from llama_wrapper import LlamaWrapper
from language_translator import LangTranslate
from custom_decorators import log_runtime_metrics

app = Flask(__name__)
CORS(app)

# app = FastAPI()

# origins = [
#     "http://127.0.0.1:8000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


#model_name = 'TheBloke/CodeLlama-7B-Instruct-GGUF'
model_name = 'TheBloke/Mistral-7B-v0.1-GGUF/mistral-7b-v0.1.Q5_K_M.gguf'
# model_name = 'TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_K_M.gguf'
# model_name = 'TheBloke/Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q4_K_M.gguf'

llama_wrapper = LlamaWrapper(model_name)
#lang_translate = LangTranslate()

class Item(BaseModel):
    input_texts: list
    stop_sequences: str
    temperature: float = 0.7
    logprobs: float = None

@app.route('/generate', methods=['POST'])
@log_runtime_metrics
def generate_text():
    data = request.get_json()

    # Get input text from the request
    input_texts = data.get('text', [])

    # Get stop sequences from the request
    stop = data.get('stop_sequences')

    temperature = data.get('temperature', 0.7)
    logprobs = data.get('logprobs')

    try:
        response = llama_wrapper.generate_text(input_texts, temperature, stop)
        print (response)
    except Exception as e:
        print (f'Error {str(e)}')
        return jsonify({"description":"An error occurred during text generation."}), 400

    return jsonify(response)

# @app.route('/generate', methods=['POST'])
@app.post('/generate2')
@log_runtime_metrics
def generate_text2(data: Item):
    # data = request.get_json()

    # Get input text from the request
    # input_texts = data.get('text', [])

    # Get stop sequences from the request
    # stop = data.get('stop_sequences')

    # temperature = data.get('temperature', 0.7)
    # logprobs = data.get('logprobs')

    input_texts = data.input_texts
    stop = data.stop_sequences

    temperature = data.temperature
    logprobs = data.logprobs

    try:
        response = llama_wrapper.generate_text(input_texts, temperature, stop)
        print (response)
    except Exception as e:
        print (f'Error {str(e)}')
        return jsonify({"description":"An error occurred during text generation."}), 400

    return jsonify(response)

@app.route('/translate', methods=['POST'])
@log_runtime_metrics
def translate_text():
    data = request.get_json()

    # Get input text from the request
    input_texts = data.get('text', '')
    src_lang = data.get('src_lang', [])
    tgt_lang = data.get('tgt_lang', [])

    
    try:
        response = lang_translate.translate_paragraph(input_texts, src_lang, tgt_lang)
    except Exception as e:
        print (f'Error {str(e)}')
        print (traceback.format_exc())
        return jsonify({"description":"An error occurred during text translation."}), 400

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)