from transformers import Pipeline, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import SUPPORTED_TASKS

import bentoml
import torch


class DialogptPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, text):
        new_user_input_ids = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors='pt')
        return new_user_input_ids

    def _forward(self, inputs):
        chat_history_ids = torch.zeros(size=(1,0), dtype=int)
        bot_input_ids = torch.cat([chat_history_ids, inputs], dim=-1)
        chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        output = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return output

    def postprocess(self, model_outputs):
        return model_outputs
    

TASK_NAME = "dialogpt-task"
TASK_DEFINITION = {
    "impl": DialogptPipeline,
    "tf": (),
    "pt": (AutoModelForCausalLM,),
    "default": {},
    "type": "text",
}
SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

chat = pipeline(
    task=TASK_NAME,
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium"),
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium"),
)

bentoml.transformers.save_model(
    "dialogpt_model",
    pipeline=chat,
    task_name=TASK_NAME,
    task_definition=TASK_DEFINITION,
)
