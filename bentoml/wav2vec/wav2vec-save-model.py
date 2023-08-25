from transformers import Pipeline, pipeline
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers.pipelines import SUPPORTED_TASKS

import bentoml
import torch
import numpy as np
import io
import soundfile as sf


class Wav2VecPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        return {}, {}, {}

    def preprocess(self, wavData):
        data, _ = sf.read(io.BytesIO(wavData))
        sample = np.array(data, dtype=float)
        return sample

    def _forward(self, sample):
        input_values = self.processor(sample, return_tensors="pt", padding="longest").input_values  # Batch size 1
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        output = self.processor.batch_decode(predicted_ids)
        return output[0]

    def postprocess(self, model_outputs):
        return model_outputs
    

TASK_NAME = "wav2vec-task"
TASK_DEFINITION = {
    "impl": Wav2VecPipeline,
    "tf": (),
    "pt": (Wav2Vec2ForCTC,),
    "default": {},
    "type": "audio",
}
SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

classifier = pipeline(
    task=TASK_NAME,
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h"),
    feature_extractor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h"),
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h"),
)

bentoml.transformers.save_model(
    "wav2vec_model",
    pipeline=classifier,
    task_name=TASK_NAME,
    task_definition=TASK_DEFINITION,
)
