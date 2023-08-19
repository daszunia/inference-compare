
from abc import ABC
import json
import logging
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))
        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes. 
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)
        new_user_input_ids = self.tokenizer.encode(sentences + self.tokenizer.eos_token, return_tensors='pt')
        return new_user_input_ids

    def inference(self, inputs):
        chat_history_ids = torch.zeros(size=(1,0), dtype=int)
        bot_input_ids = torch.cat([chat_history_ids, inputs], dim=-1)
        chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        output = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        return [output]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
