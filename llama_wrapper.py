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
# from ctransformers import LLM
from tokenizers import Tokenizer

model_dir = '/Users/saurav/.cache/lm-studio/models/'

from ctypes import (
    c_int,
)
from typing import (
    List,
    Optional,
    Sequence,
    Union,
)

class CustomLLM:
    def __init__(self, llm: Optional[AutoModelForCausalLM] = None):
        self.tokenizer = Tokenizer.from_file("../llm_finetuning/tokenizer/my_bpe_tokenizer.json")
        self.custom_tokenizer_fn = llm.tokenize
        self.custom_detokenizer_fn = llm.detokenize

    def tokenize(self, text: str, add_bos_token: Optional[bool] = None) -> List[int]:
        tokens = self.tokenizer.encode(text).ids
        print (self.custom_tokenizer_fn(text, add_bos_token))
        print (tokens)
        print (self.custom_detokenizer_fn(tokens))
        return tokens
        """Converts a text into list of tokens.

        Args:
            text: The text to tokenize.
            add_bos_token: Whether to add the beginning-of-sequence token.

        Returns:
            The list of tokens.
        """
        if add_bos_token is None:
            add_bos_token = self.model_type == "llama"
        tokens = (c_int * (len(text) + 1))()
        n_tokens = self.ctransformers_llm_tokenize(text.encode(), add_bos_token, tokens)
        return tokens[:n_tokens]

    def detokenize(
        self,
        tokens: Sequence[int],
        decode: bool = True,
    ) -> Union[str, bytes]:
        """Converts a list of tokens to text.

        Args:
            tokens: The list of tokens.
            decode: Whether to decode the text as UTF-8 string.

        Returns:
            The combined text of all tokens.
        """
        if isinstance(tokens, int):
            tokens = [tokens]
        texts = self.tokenizer.decode(tokens)
        # texts = []
        # for token in tokens:
        #     text = self.ctransformers_llm_detokenize(token)
        #     texts.append(text)
        texts = "".join(texts)
        texts = texts.encode()
        if decode:
            texts = texts.decode(errors="ignore")
            # https://github.com/ggerganov/llama.cpp/blob/43033b7bb4858da4f591715b3babdf906c9b7cbc/common/common.cpp#L778-L781
            if tokens[:1] == [self.bos_token_id] and texts[:1] == " ":
                texts = texts[1:]
        return texts

class LlamaWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        model_path = os.path.join(model_dir, model_name)
        # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama", gpu_layers=100, context_length = 8192)
        self.custom_llm = CustomLLM(self.llm)
        self.llm.tokenize = self.custom_llm.tokenize
        # self.llm.detokenize = self.custom_llm.detokenize

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

