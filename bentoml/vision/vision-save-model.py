from transformers import Pipeline, pipeline
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers.pipelines import SUPPORTED_TASKS

import bentoml
import requests
from PIL import Image


class VisionPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, imageURL):
        image = Image.open(requests.get(imageURL, stream=True).raw)
        return image

    def _forward(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        output = self.model.config.id2label[predicted_class_idx]
        return output

    def postprocess(self, model_outputs):
        return model_outputs
    

TASK_NAME = "vis-task"
TASK_DEFINITION = {
    "impl": VisionPipeline,
    "tf": (),
    "pt": (ViTForImageClassification,),
    "default": {},
    "type": "text",
}
SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

classifier = pipeline(
    task=TASK_NAME,
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'),
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'),
)

bentoml.transformers.save_model(
    "vis_model",
    pipeline=classifier,
    task_name=TASK_NAME,
    task_definition=TASK_DEFINITION,
)
