import kserve
from typing import Dict
import torch
import requests
from PIL import Image

from transformers import ViTImageProcessor, ViTForImageClassification


class KServe_Vision(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.tokenizer = None
        self.chat_history_ids = torch.zeros(size=(1,0), dtype=int)  # Empty chat history -> empty tensor of shape (1,0)
        self.load()

    def load(self):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.ready = True

    def predict(self, request, _) -> Dict:
        imageURL = request.decode("utf-8").strip()
        image = Image.open(requests.get(imageURL, stream=True).raw)

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        output = self.model.config.id2label[predicted_class_idx]

        return {"result": output}


if __name__ == "__main__":
    model = KServe_Vision("kserve-custom-vision")
    kserve.ModelServer().start([model])