import kserve
from typing import Dict
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


class KServe_Dialogpt(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.tokenizer = None
        self.chat_history_ids = torch.zeros(size=(1,0), dtype=int)  # Empty chat history -> empty tensor of shape (1,0)
        self.load()

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.ready = True

    def predict(self, request, _) -> Dict:
        jsonStr = request.decode("utf-8")
        jsonObj = json.loads(jsonStr)

        # Input follows the Tensorflow V1 HTTP API for binary values
        # https://www.tensorflow.org/tfx/serving/api_rest#encoding_binary_values
        input = jsonObj["data"]
        inputs = self.tokenizer.encode(input + self.tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([self.chat_history_ids, inputs], dim=-1)
        chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        output = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        return {"result": output}


if __name__ == "__main__":
    model = KServe_Dialogpt("kserve-custom-dialogpt")
    kserve.ModelServer().start([model])