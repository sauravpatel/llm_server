import os
import re
import time
import json

"""
# Base ctransformers with no GPU acceleration
pip install ctransformers>=0.2.24
# Or with CUDA GPU acceleration
pip install ctransformers[cuda]>=0.2.24
# Or with ROCm GPU acceleration
CT_HIPBLAS=1 pip install ctransformers>=0.2.24 --no-binary ctransformers
# Or with Metal GPU acceleration for macOS systems
CT_METAL=1 pip install ctransformers>=0.2.24 --no-binary ctransformers
"""
from ctransformers import AutoModelForCausalLM

model_dir = '/Users/saurav/.cache/lm-studio/models/'

class LlamaWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        model_path = os.path.join(model_dir, model_name)
        # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama", gpu_layers=100, context_length = 8192)

    def trim_text_at_stop_sequences(self, stop_sequences, input_text):
        # Create a regular expression pattern using alternation (|) for stop sequences
        pattern = '|'.join(map(re.escape, stop_sequences))

        # Use re.split to split the text at the first occurrence of any stop sequence
        parts = re.split(pattern, input_text, 1)

        # Return the trimmed text
        trimmed = parts[0].strip()

        print (f'Trimmed {trimmed} -- Actual {input_text}')

        return trimmed

    def generate_text(self, input_texts, temperature = .8, stop_sequences = None):
        if not isinstance(input_texts, list):
            input_texts = [input_texts]

        # Measure the start time
        start_time = time.time()

        print (input_texts)

        generated_texts = [self.llm(input_text, stop=stop_sequences, max_new_tokens=4096) for input_text in input_texts]

        print (f'OUTPUT {json.dumps(generated_texts, indent = 4)}')

        # Measure the end time
        end_time = time.time()

        # Calculate the runtime
        runtime = end_time - start_time

        # Format the response as per the specified output format
        response_data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": i,
                    "logprobs": None,
                    "text": generated_text
                } for i, generated_text in enumerate(generated_texts)
            ],
            "created": int(time.time()),
            "id": f"chatcmpl-{hash(''.join(generated_texts))}",
            "model": self.model_name,
            "object": "chat.completion",
            "usage": {
                "completion_tokens": sum(len(text.split()) for text in generated_texts),
                "prompt_tokens": sum(len(text.split()) for text in input_texts),
                "total_tokens": sum(len(text.split()) for text in generated_texts) + sum(len(text.split()) for text in input_texts)
            }
        }

        return response_data

