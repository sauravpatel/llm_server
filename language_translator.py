import json

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class LangTranslate:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=False)#, src_lang="tel_Telu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=False)
        self.lang_mapping = {
            'te_IN': 'tel_Telu',
            'en_XX': 'eng_Latn'
        }
        #self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        #self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        #self.lang_mapping = {}

    def translate_single(self, input_text, src_lang, tgt_lang):
        src_lang = self.lang_mapping.get(src_lang, src_lang)
        tgt_lang = self.lang_mapping.get(tgt_lang, tgt_lang)

        print(src_lang, tgt_lang)

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(input_text, return_tensors="pt")

        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang], max_length=30
        )
        generated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return generated_text

    def translate_single2(self, input_text, src_lang, tgt_lang):
        self.tokenizer.src_lang = src_lang
        encoded_hi = self.tokenizer(input_text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_hi,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang]
        )
        generated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return generated_text
        # => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

    def translate_paragraph(self, input_string, src_lang, tgt_lang):
        input_texts = input_string.split('.')
        for input_text in input_texts:
            print (f'{input_text}\n')
        return [self.translate_single(input_text, src_lang, tgt_lang) for input_text in input_texts]
