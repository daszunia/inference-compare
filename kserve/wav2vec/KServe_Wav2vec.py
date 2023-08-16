import kserve
from typing import Dict
import torch
import numpy as np
import io
import soundfile as sf

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class KServe_Wav2vec(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.tokenizer = None
        self.chat_history_ids = torch.zeros(size=(1,0), dtype=int)  # Empty chat history -> empty tensor of shape (1,0)
        self.load()

    def load(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.ready = True

    def predict(self, request, _) -> Dict:
        print(request)
        print(type(request))
        data, _ = sf.read(io.BytesIO(request))
        sample = np.array(data, dtype=float)

        input_values = self.processor(sample, return_tensors="pt", padding="longest").input_values  # Batch size 1
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        output = self.processor.batch_decode(predicted_ids)

        return {"result": output}


if __name__ == "__main__":
    model = KServe_Wav2vec("kserve-custom-wav2vec")
    kserve.ModelServer().start([model])